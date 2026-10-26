#!/usr/bin/env python3
"""gemini-cloud-agent (24/7 edition).

Runs on a Hugging Face Docker Space. Exposes an authenticated HTTP API that lets
a Gemini API instance execute bash on this container, plus a tiny browser console.
State is mirrored to a private HF dataset so it survives container restarts.
"""
import hmac
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from huggingface_hub import HfApi, hf_hub_download

AUTH = os.environ.get("AGENT_TOKEN", "").strip()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
DATA_REPO = os.environ.get("DATA_REPO", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
PORT = int(os.environ.get("PORT", "7860"))
WORK = os.path.expanduser("~/data")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

os.makedirs(WORK, exist_ok=True)
BOOT = time.time()
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

SYSTEM = (
    "You are Gemini operating as an autonomous Linux operator on a cloud container "
    "that runs 24/7 and that you fully control. You have exactly one tool, run_shell, "
    "which executes bash in your workspace and returns stdout, stderr and exit code. "
    "Work in small verifiable steps, read every output before the next command, fix "
    "errors yourself, never invent output, and finish with a plain statement of what "
    "you ran plus the evidence."
)
TOOLS = [{
    "functionDeclarations": [{
        "name": "run_shell",
        "description": "Execute a bash command in the cloud workspace and return stdout, stderr, exit code.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {"type": "STRING", "description": "Bash command"},
                "timeout": {"type": "INTEGER", "description": "Seconds, default 120"},
            },
            "required": ["command"],
        },
    }]
}]

JOBS = {}
JOBS_LOCK = threading.Lock()


def run_shell(command, timeout=120):
    try:
        p = subprocess.run(["bash", "-c", command], cwd=WORK, capture_output=True,
                           text=True, timeout=int(timeout))
        return {"exit_code": p.returncode, "stdout": p.stdout[-8000:],
                "stderr": p.stderr[-4000:], "cwd": WORK, "host": os.uname().nodename}
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "stdout": "", "stderr": "killed: timeout %ss" % timeout,
                "cwd": WORK, "host": os.uname().nodename}
    except Exception as exc:  # noqa: BLE001
        return {"exit_code": 1, "stdout": "", "stderr": repr(exc), "cwd": WORK,
                "host": os.uname().nodename}


def gemini_call(contents, attempt=0):
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": contents,
        "tools": TOOLS,
        "generationConfig": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(API.format(model=MODEL), data=payload,
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": GEMINI_KEY})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None)
        if code in (429, 500, 502, 503, 504) and attempt < 6:
            wait = min(90, 6 * (2 ** attempt))
            time.sleep(wait)
            return gemini_call(contents, attempt + 1)
        raise


def run_task(prompt, max_turns=14):
    log = []
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    log.append("[agent] model=%s host=%s" % (MODEL, os.uname().nodename))
    for _ in range(max_turns):
        time.sleep(1.5)
        data = gemini_call(contents)
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
        contents.append({"role": "model", "parts": parts})
        text = "".join(p.get("text", "") for p in parts
                       if p.get("text") and not p.get("thought"))
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if text.strip():
            log.append("[gemini] " + text.strip())
        if not calls:
            return {"status": "done", "log": log, "final": text.strip()}
        replies = []
        for call in calls:
            args = call.get("args") or {}
            cmd = args.get("command", "")
            log.append("[gemini -> run_shell] " + cmd)
            res = run_shell(cmd, args.get("timeout", 120))
            tail = (res["stdout"] or res["stderr"]).strip().splitlines()
            log.append("[host] exit=%s  %s" % (res["exit_code"], tail[-1][:300] if tail else ""))
            replies.append({"functionResponse": {"name": call.get("name"), "response": res}})
        contents.append({"role": "user", "parts": replies})
    return {"status": "max_turns", "log": log, "final": ""}


def job_worker(job_id, prompt):
    try:
        result = run_task(prompt)
    except Exception:  # noqa: BLE001
        result = {"status": "error", "log": [traceback.format_exc()[-2000:]], "final": ""}
    with JOBS_LOCK:
        JOBS[job_id].update(result)


def sync_down():
    if not (api and DATA_REPO):
        return
    try:
        files = api.list_repo_files(DATA_REPO, repo_type="dataset")
        for path in files:
            try:
                local = hf_hub_download(DATA_REPO, path, repo_type="dataset", token=HF_TOKEN)
                dest = os.path.join(WORK, path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest) or os.path.getmtime(local) > os.path.getmtime(dest):
                    with open(local, "rb") as src, open(dest, "wb") as dst:
                        dst.write(src.read())
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def sync_up():
    if not (api and DATA_REPO):
        return
    try:
        api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=WORK, repo_id=DATA_REPO, repo_type="dataset",
                          commit_message="sync @ %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
                          ignore_patterns=["*.pyc", "__pycache__/*", ".cache/*"])
    except Exception:  # noqa: BLE001
        pass


def sync_loop():
    while True:
        time.sleep(300)
        sync_up()


def keepalive_loop():
    host = (os.environ.get("SPACE_HOST", "").strip()
            or os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip())
    if not host:
        return
    url = "https://%s/health" % host
    while True:
        time.sleep(600)
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + AUTH})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:  # noqa: BLE001
            pass


CONSOLE = """<!doctype html><meta charset=utf-8><title>gemini-cloud-agent</title>
<style>body{font:14px ui-monospace,monospace;background:#0b0f14;color:#c9d1d9;margin:0;padding:24px}
input,button,textarea{font:inherit;background:#111820;color:#c9d1d9;border:1px solid #263241;border-radius:6px;padding:8px}
#out{white-space:pre-wrap;border:1px solid #263241;border-radius:8px;padding:12px;margin-top:12px;min-height:220px}</style>
<h3>gemini-cloud-agent <span id=s style=color:#888>idle</span></h3>
<p>token <input id=t size=44> &nbsp; prompt <input id=p size=52 value="report host and uptime">
<button onclick=go()>run</button> <button onclick=sh()>shell</button></p>
<div id=out></div>
<script>
let tok=()=>document.getElementById('t').value;
async function j(u,b){let r=await fetch(u,{method:'POST',headers:{'Authorization':'Bearer '+tok(),'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
async function sh(){let c=document.getElementById('p').value;document.getElementById('s').textContent='running';
 let r=await j('/exec',{cmd:c});document.getElementById('out').textContent=(r.stdout||'')+(r.stderr?'\\n[stderr]\\n'+r.stderr:'')+'\\n[exit] '+r.exit_code;document.getElementById('s').textContent='idle'}
async function go(){document.getElementById('s').textContent='queued';
 let r=await j('/task',{prompt:document.getElementById('p').value});let id=r.job_id;
 if(!id){document.getElementById('out').textContent=JSON.stringify(r);return}
 document.getElementById('s').textContent='running';
 for(;;){await new Promise(z=>setTimeout(z,3000));
  let s=await fetch('/job/'+id,{headers:{'Authorization':'Bearer '+tok()}}).then(x=>x.json());
  document.getElementById('out').textContent=(s.log||[]).join('\\n');
  if(s.status!=='running'){document.getElementById('s').textContent=s.status;break}}}
</script>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "gemini-cloud-agent/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        got = (self.headers.get("Authorization") or "").strip()
        want = "Bearer " + AUTH
        return bool(AUTH) and hmac.compare_digest(got, want)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            return {}

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "host": os.uname().nodename,
                                    "uptime_s": int(time.time() - BOOT),
                                    "model": MODEL, "workspace": WORK,
                                    "persistence": bool(api and DATA_REPO)})
        if self.path == "/":
            return self._send(200, CONSOLE, "text/html; charset=utf-8")
        if self.path.startswith("/job/"):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            job = JOBS.get(self.path[5:])
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, job)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/exec":
            b = self._body()
            return self._send(200, run_shell(b.get("cmd", ""), b.get("timeout", 120)))
        if self.path == "/task":
            b = self._body()
            job_id = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "running", "log": [], "final": ""}
            threading.Thread(target=job_worker, args=(job_id, b.get("prompt", "")),
                             daemon=True).start()
            return self._send(200, {"job_id": job_id})
        if self.path == "/sync":
            threading.Thread(target=sync_up, daemon=True).start()
            return self._send(200, {"status": "sync started"})
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    threading.Thread(target=sync_down, daemon=True).start()
    if api and DATA_REPO:
        threading.Thread(target=sync_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    print("listening on %s" % PORT, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

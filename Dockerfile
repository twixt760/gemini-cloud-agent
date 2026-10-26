FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends bash curl git coreutils procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "huggingface_hub>=0.34"

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 7860
CMD ["python", "/app/app.py"]

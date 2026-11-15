# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.11-slim

FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline --no-audit --progress=false
COPY frontend ./
RUN npm run build

FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AMADEUS_CONFIG=/app/config/config.yaml \
    KURISU_FRONTEND_DIR=/app/frontend_dist

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY amadeus ./amadeus
COPY config ./config
COPY assets ./assets
COPY static ./static
COPY --from=frontend-builder /frontend/dist ./frontend_dist

RUN pip install --upgrade pip \
    && pip install ".[web,gemini]" gradio-client

RUN cp /app/config/config.fly.yaml /app/config/config.yaml \
    && mkdir -p logs transcripts

EXPOSE 8080

CMD ["uvicorn", "amadeus.web.app:app", "--host", "0.0.0.0", "--port", "8080"]

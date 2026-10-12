FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        poppler-utils \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tesseract est multi-thread (OpenMP) : sans limite, chaque instance prend tous les
# cœurs. La réindexation en lance plusieurs en parallèle (MATCHING_REINDEX_DOWNLOAD_WORKERS)
# et, mesuré sur le VPS (4 cœurs), 8 OCR simultanés saturaient la machine (load 32) au
# point de dépasser le délai de 25 s par page, alors que le même lot passe en 2,4 s avec
# un thread chacun. La limite est posée sur Tesseract SEUL, par ce script d'enrobage
# (app/extraction.py le sélectionne) : un OMP_THREAD_LIMIT global limiterait aussi
# PyTorch (embeddings, cross-encoder) à un thread.
RUN printf '#!/bin/sh\nexport OMP_THREAD_LIMIT=1\nexec tesseract "$@"\n' > /usr/local/bin/tesseract-1t \
    && chmod +x /usr/local/bin/tesseract-1t

WORKDIR /app

COPY requirements.txt ./
ENV PIP_DEFAULT_TIMEOUT=600
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

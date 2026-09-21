FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        openjdk-21-jre-headless \
        tesseract-ocr \
        poppler-utils \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
ENV PIP_DEFAULT_TIMEOUT=600
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

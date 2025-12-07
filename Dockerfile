FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader stopwords wordnet punkt -d /app/nltk_data
ENV NLTK_DATA=/app/nltk_data

EXPOSE 5050

CMD ["python", "app.py"]
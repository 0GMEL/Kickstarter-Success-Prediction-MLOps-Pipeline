FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader stopwords wordnet punkt -d /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

EXPOSE 5050

CMD ["python", "app.py"]

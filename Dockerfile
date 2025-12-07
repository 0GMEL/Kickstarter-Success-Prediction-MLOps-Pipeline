FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader stopwords wordnet punkt -d /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

EXPOSE 5050

CMD ["python", "app.py"]
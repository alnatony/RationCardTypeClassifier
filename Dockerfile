FROM python:3.11-slim

RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 10000
CMD ["streamlit", "run", "src/app/app1.py", "--server.port", "10000", "--server.address", "0.0.0.0"]
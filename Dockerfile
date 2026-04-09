FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data/knowledge_bench_public.csv ./data/knowledge_bench_public.csv

CMD ["python", "src/run_public_inference.py"]

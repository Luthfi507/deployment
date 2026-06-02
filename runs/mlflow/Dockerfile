FROM python:3.11-slim

WORKDIR /mlflow

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:///mlflow/artifacts
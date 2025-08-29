FROM python:3.10-slim

WORKDIR /app
RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", 
     "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow-db:5432/mlflow",
     "--default-artifact-root", "/mlflow/artifacts",
     "--host", "0.0.0.0",
     "--port", "5000"]
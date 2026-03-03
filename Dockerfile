FROM python:3.13-slim

# Evita arquivos .pyc e melhora logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY src ./src
COPY app ./app
COPY artifacts ./artifacts
COPY data ./data
COPY pytest.ini ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Built and run-tested locally with Docker Desktop 29.9.3 (see docs/DECISIONS.md, D17).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first, so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY spamclf/ spamclf/
COPY models/ models/

# Do not run as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Hosts such as Render set $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn spamclf.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

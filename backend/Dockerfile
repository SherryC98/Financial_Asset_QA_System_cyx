FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage (no build-essential, smaller image) ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Reduce memory: disable bytecode caching, enable GC optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=65536

COPY . .

EXPOSE 8001

CMD ["python", "-c", "import os; port = os.environ.get('PORT', '8001'); os.execvp('python', ['python', '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', port])"]

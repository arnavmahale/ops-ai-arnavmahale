# Multi-stage Dockerfile
# Stage 1: Builder (installs dependencies, large layer)
# Stage 2: Runtime (copy only what's needed, smaller final image)
# Why? Faster deployments, smaller disk usage in container registry

FROM python:3.11-slim as builder
WORKDIR /app
COPY week2/backend/requirements.txt .

# Install dependencies + system libraries
# Important: LightGBM needs libgomp1 (OpenMP library)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - minimal image
FROM python:3.11-slim
WORKDIR /app

# Install only runtime dependencies (NOT build tools)
RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Copy pre-installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY week2/backend/ .
COPY week2/metadata /metadata
RUN mkdir -p /data/processed /app/frontend/public

# Health check: Kubernetes will call this periodically
# Goal: verify model is loaded and can handle requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start FastAPI server on 0.0.0.0:8000 (accessible from outside container)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

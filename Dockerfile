# Multi-stage Python 3.11 Dockerfile for Apollo MCP Server
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code and config
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    APOLLO_TRANSPORT=sse \
    APOLLO_HOST=0.0.0.0 \
    APOLLO_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/sse || exit 0

CMD ["python", "-m", "apollo.main", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]


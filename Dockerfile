# Dockerfile for FastAPI application
FROM python:3.9-slim-bullseye

ENV PIP_PROGRESS_BAR=off \
    PIP_NO_PROGRESS_BAR=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    GOTO_NUM_THREADS=1

# Set working directory
WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel --progress-bar off

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install \
    --prefer-binary \
    --no-cache-dir \
    --disable-pip-version-check \
    --progress-bar off \
    -r requirements.txt

# Copy application code
COPY . .

# Copy wait-for-it script explicitly for old docker-compose v2 startup flow
COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, httpx; port = os.getenv('PORT', '8080'); ssl_enabled = os.getenv('SSL_ENABLED', 'false').lower() == 'true'; scheme = 'https' if ssl_enabled else 'http'; url = f'{scheme}://localhost:{port}/'; kwargs = {'verify': False} if ssl_enabled else {}; httpx.get(url, timeout=5.0, **kwargs)" || exit 1

# Run application
CMD ["python", "main.py"]

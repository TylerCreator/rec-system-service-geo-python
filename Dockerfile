# Dockerfile for FastAPI application
FROM python:3.9-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy wait-for-it script explicitly for old docker-compose v2 startup flow
COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, httpx; port = os.getenv('PORT', '8080'); ssl_enabled = os.getenv('SSL_ENABLED', 'false').lower() == 'true'; scheme = 'https' if ssl_enabled else 'http'; url = f'{scheme}://localhost:{port}/'; kwargs = {'verify': False} if ssl_enabled else {}; httpx.get(url, timeout=5.0, **kwargs)" || exit 1

# Run application
CMD ["python", "main.py"]

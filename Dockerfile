FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STREAMLIT_PORT=8501 \
    ENVIRONMENT=production

# Install system dependencies (added curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/app/data/uploads /app/app/data/vector_store

# Streamlit config (to avoid CORS/XSRF issues)
RUN mkdir -p /root/.streamlit && \
    echo "[server]\n\
    enableCORS = false\n\
    enableXsrfProtection = false\n\
    " > /root/.streamlit/config.toml

# Expose ports
EXPOSE $PORT $STREAMLIT_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run both FastAPI & Streamlit
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port $PORT & \
           streamlit run streamlit_app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0"

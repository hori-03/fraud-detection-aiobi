# Use Python 3.11 slim image
FROM python:3.11-slim

# Build version to force cache invalidation
ENV BUILD_VERSION=2.1.1-s3-upload-fix
ENV BUILD_DATE=2025-11-07

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY APP_autoML/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from APP_autoML
COPY APP_autoML/ .

# Copy automl_transformer from parent directory (important for full_automl.py)
COPY automl_transformer/ /automl_transformer

# Copy utils from parent directory (needed by automl_transformer)
COPY utils/ /utils

# Copy data/metatransformer_training (model weights)
COPY data/metatransformer_training/ /data/metatransformer_training

# Create necessary directories
RUN mkdir -p uploads logs models/xgboost_models instance

# Set environment variables
ENV FLASK_APP=run.py
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/:/automl_transformer:/utils
ENV PORT=8080

# Expose port (Railway will set PORT env variable)
EXPOSE 8080

# Run with gunicorn for production
# Use shell form to allow variable expansion
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 2 --timeout 300 --keep-alive 5 run:app

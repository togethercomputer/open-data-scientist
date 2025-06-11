# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create custom_data directory
RUN mkdir -p /app/custom_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub
RUN pip install --no-cache-dir sentence-transformers transformers torch accelerate

# Copy the application code
COPY . .

# Debug: Create the target directory explicitly first
RUN mkdir -p /app/downloaded_data && echo "Debug: Explicitly created /app/downloaded_data"

# Download data
RUN python download_data.py

# Debug: Investigate filesystem state immediately after download_data.py
RUN echo "Debug: Listing /app contents after download script:" && ls -lA /app
RUN echo "Debug: Listing /app/downloaded_data contents recursively after download script:" && ls -lAR /app/downloaded_data || echo "Debug: /app/downloaded_data not found or ls failed"
RUN echo "Debug: Attempting to find payments.csv globally after download script:" && find / -name payments.csv -type f -print -quit || echo "Debug: payments.csv not found globally by find"
RUN echo "Debug: Checking current working directory during build:" && pwd

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"] 
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY run_pipeline.py ./
COPY app.py ./
COPY requirements.txt ./

# Create directories for outputs
RUN mkdir -p data/processed reports/figures

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["python", "run_pipeline.py"]
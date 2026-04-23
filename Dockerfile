# Use Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Also install xgboost explicitly
RUN pip install --no-cache-dir xgboost scikit-learn

# Copy app files
COPY . .

# Expose port
ENV PORT=5000

# Run the app
CMD ["python", "app.py"]

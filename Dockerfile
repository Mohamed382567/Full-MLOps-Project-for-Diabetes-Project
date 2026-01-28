FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- CRITICAL FIX: Ensure logs are visible in Render real-time ---
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solve the HuggingFace/Gradio conflict
RUN pip install --no-cache-dir "huggingface_hub<0.26.0"
RUN pip install --no-cache-dir gradio==4.19.1

# Copy everything else
COPY . .

# Ensure the directory exists
RUN mkdir -p diabetes-model-artifacts

# Set port environment
ENV PORT=8000
EXPOSE 8000

# --- OPTIMIZED STARTUP ---
# Reduced sleep to 5 seconds to prevent Render port-scan timeout
# Runs API in background and Gradio in foreground
CMD python src/app/main.py & sleep 5 && python src/app/gradio_app.py

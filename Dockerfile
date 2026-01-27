FROM python:3.9-slim

# Install system dependencies (Updated package names for newer Debian)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX: Solve the HuggingFace/Gradio conflict ---
# We force a specific version of huggingface_hub that contains HfFolder
RUN pip install --no-cache-dir "huggingface_hub<0.26.0"
RUN pip install --no-cache-dir gradio==4.19.1

# Copy everything else
COPY . .

RUN mkdir -p artifacts

ENV PORT=8000
EXPOSE $PORT

CMD ["python", "src/app/gradio_app.py"]

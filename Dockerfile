FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (only what the app needs)
RUN pip install --no-cache-dir \
    gradio>=4.0.0 \
    matplotlib>=3.7.0 \
    numpy>=1.24.0 \
    onnxruntime>=1.16.0 \
    Pillow>=10.0.0 \
    PyYAML>=6.0 \
    huggingface_hub>=0.20.0

# Copy application files
COPY app.py .
COPY params.yaml .
COPY src/common.py src/common.py
COPY examples/ examples/

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "app.py"]

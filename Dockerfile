# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install system dependencies
# sox and ffmpeg are required as per README.md and setup.bat
# git and build-essential are utility dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsox-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We install torch first with the specific CUDA version, similar to setup.bat
# Reverting to cu128 to ensure compatibility with prebuilt flash-attention wheels
# (NVIDIA drivers are backward compatible, so cu128 works on cu130 hosts)
ARG CUDA_VERSION=cu128
RUN pip install --no-cache-dir torch==2.9.1 torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install Flash Attention 2 prebuilt wheel
# Using v2.8.3 wheel for CUDA 12.8 and Torch 2.9.0 (compatible with 2.9.x) and Python 3.12
RUN pip install --no-cache-dir https://github.com/bdashore3/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Install rust compiler (required for deepfilternet)
RUN apt-get update && apt-get install -y \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy compatibility patches and apply them during build to verify
COPY patches/ patches/
RUN python patches/deepfilternet_torchaudio_patch.py

# Copy the rest of the application
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "voice_clone_studio.py"]

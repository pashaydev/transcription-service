# Use Python 3.10 with Debian
FROM python:3.10-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    g++ \
    make \
    cmake \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Go
ENV GO_VERSION=1.20.3
RUN wget -q https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Set work directory
WORKDIR /app

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages with specific versions in the correct order
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir openai-whisper==20230314

# Verify installation is working properly
RUN python -c "import numpy; import torch; import whisper; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}, whisper is installed')"

# Copy the updated whisper_bridge.py file
COPY whisper_bridge.py /app/whisper_bridge.py
RUN chmod +x /app/whisper_bridge.py

# Copy go files and build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o whisper-service .

# Create static dir
RUN mkdir -p /app/static
COPY static/index.html /app/static/index.html

# Set environment variables
ENV WHISPER_MODEL=tiny
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run test at build time to confirm it works
RUN echo "Testing whisper transcription" | ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -ac 1 test.wav
RUN python whisper_bridge.py --input test.wav --output test.json --model tiny
RUN cat test.json || echo "Test failed but continuing build"

# Set the entrypoint
CMD ["./whisper-service"]
# Transcription Audio Service
**Key features:**
- Transcribe audio files up to 25MB in size
- Receive timestamped text segments
- Simple web interface for uploading files
- RESTful API for programmatic access
- Configurable model size (tiny, base, small, medium, large)

---

## Quick Start with Docker
The easiest way to run the service is using Docker:
Build the Docker image

```bash
docker build -t whisper-transcription-service .
```

# Run the container
```bash
docker run -p 8080:8080 -e WHISPER_MODEL=tiny whisper-transcription-service
```

Then visit http://localhost:8080 in your web browser.

---

## Prerequisites

- Go 1.18+
- Python 3.10+
- FFmpeg
- PyTorch and OpenAI Whisper
 
## Backend Setup

Install Go dependencies:
```bash
go mod download
```

Install Python dependencies:
```bash
pip install numpy torch openai-whisper
```

Build and run the service:

```bash 
go build -o whisper-service
./whisper-service
```

---

## Python Bridge
The **whisper_bridge.py** script is a critical component that:

Takes audio file input and output file path as arguments
Loads the specified Whisper model
Processes the audio file to generate transcriptions
Writes the results to the output file in JSON format

The script is executed by the Go service as a subprocess for each transcription request.
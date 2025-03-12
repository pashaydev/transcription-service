#!/usr/bin/env python3
import sys
import json
import os
import traceback
import argparse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stderr)
logger = logging.getLogger('whisper_bridge')

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using whisper")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--model", "-m", default="tiny", help="Whisper model to use")
    args = parser.parse_args()

    start_time = time.time()

    try:
        # Import numpy first to verify it's available
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")

        # Import torch to verify it's available
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")

        # Now import whisper
        import whisper

        # Check if the input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file does not exist: {args.input}")
            with open(args.output, "w") as f:
                json.dump({
                    "error": f"Input file does not exist: {args.input}",
                    "segments": []
                }, f, indent=2)
            return 1

        # Check file size
        file_size_mb = os.path.getsize(args.input) / (1024 * 1024)
        logger.info(f"Input file size: {file_size_mb:.2f} MB")

        # Load model
        logger.info(f"Loading whisper model: {args.model}")
        model = whisper.load_model(args.model, device="cpu")
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

        # Transcribe
        logger.info(f"Transcribing: {args.input}")
        result = model.transcribe(args.input, fp16=False)

        # Process segments
        segments = []
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"],
                "start_time": segment["start"],
                "end_time": segment["end"]
            })

        # Write output
        with open(args.output, "w") as f:
            json.dump({"segments": segments}, f, indent=2)

        logger.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Transcribed {len(segments)} segments")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        logger.error(traceback.format_exc())

        # Create minimal output in case of error
        with open(args.output, "w") as f:
            json.dump({
                "error": str(e),
                "segments": [{
                    "text": f"Error transcribing audio: {str(e)}",
                    "start_time": 0,
                    "end_time": 0
                }]
            }, f, indent=2)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
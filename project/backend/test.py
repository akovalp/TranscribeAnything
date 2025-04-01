#!/usr/bin/env python3
"""
Simple test script for the MLX Whisper Transcription API.
Tests transcription functionality by sending an audio file to the backend.
"""
import os
import sys
import requests
from pathlib import Path
import argparse
import json


def test_transcribe(audio_file_path: str, api_url: str = "http://localhost:8000"):
    """
    Test transcription API by sending an audio file and displaying the response.

    Args:
        audio_file_path: Path to the audio file to transcribe
        api_url: Base URL of the API (default: http://localhost:8000)
    """
    # Validate the file exists
    file_path = Path(audio_file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    # Check file extension
    allowed_extensions = [".mp3", ".wav", ".m4a", ".flac", ".mov"]
    if file_path.suffix.lower() not in allowed_extensions:
        print(
            f"Error: Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}")
        sys.exit(1)

    # Prepare the request
    url = f"{api_url}/transcribe/"

    print(f"Sending file {file_path} to {url}...")

    try:
        # Create form data
        files = {"file": (file_path.name, open(
            file_path, "rb"), f"audio/{file_path.suffix[1:]}")}
        data = {
            "language": None,  # Auto-detect language
            "word_timestamps": "true",
            "fp16": "true",
            "best_of": "5",
            "no_speech_threshold": "0.6",
            "condition_on_previous_text": "true"
        }

        # Send request
        response = requests.post(url, files=files, data=data)

        # Process response
        if response.status_code == 200:
            result = response.json()
            print("\nTranscription successful!")
            print("\nTranscribed text:")
            print("-" * 40)
            print(result["text"])
            print("-" * 40)
            print(f"\nDetected language: {result['language']}")

            if "segments" in result and result["segments"]:
                print(f"\nSegments: {len(result['segments'])}")
                # Print first segment as example
                print(
                    f"First segment: {json.dumps(result['segments'][0], indent=2)}")
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {api_url}")
        print("Make sure the server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the MLX Whisper Transcription API")
    parser.add_argument(
        "audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the API (default: http://localhost:8000)")

    args = parser.parse_args()
    test_transcribe(args.audio_file, args.api_url)

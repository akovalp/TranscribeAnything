# backend/main.py

import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from models import TranscriptionRequest, TranscriptionResponse
from utils import transcribe_audio
import uvicorn
logger = logging.getLogger(__name__)
# Create FastAPI app
app = FastAPI(
    title="MLX Whisper Transcription API",
    description="API for transcribing audio files using MLX Whisper",
    version="0.1.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint


@app.get("/", tags=["root"])
async def root():
    return {"message": "MLX Whisper Transcription API"}


@app.post("/transcribe/", response_model=TranscriptionResponse, tags=["transcription"])
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
    fp16: bool = Form(True),
    best_of: int = Form(5),
    no_speech_threshold: float = Form(0.6),
    hallucination_silence_threshold: Optional[float] = Form(None),
    condition_on_previous_text: bool = Form(True)
):
    """
    Transcribe an audio file using MLX Whisper
    """
    # Check file size (limit to 25MB for example)
    if file.size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail="File too large (max 100MB)")

    # Check file type
    allowed_extensions = [".mp3", ".wav", ".m4a", ".flac", ".mov"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Read file content
    file_content = await file.read()

    try:
        # Process the audio
        result = await transcribe_audio(
            file_content,
            language=language,
            word_timestamps=word_timestamps,
            fp16=fp16,
            best_of=best_of,
            no_speech_threshold=no_speech_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            condition_on_previous_text=condition_on_previous_text
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Transcription error: {str(e)}")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

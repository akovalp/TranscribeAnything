#backend/models.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    word_timestamps: bool = False
    fp16: bool = True
    best_of: int = 5
    no_speech_threshold: float = 0.6
    hallucination_silence_threshold: Optional[float] = None
    condition_on_previous_text: bool = True


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    segments: Optional[List[Dict[str, Any]]] = None

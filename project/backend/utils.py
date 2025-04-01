#backend/utils.py
import os
import tempfile
import mlx_whisper


async def transcribe_audio(
    file_content: bytes,
    language: str = None,
    word_timestamps: bool = False,
    fp16: bool = True,
    best_of: int = 5,
    no_speech_threshold: float = 0.6,
    hallucination_silence_threshold: float = None,
    condition_on_previous_text: bool = True
):
    """
    Transcribe audio using MLX Whisper
    """
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # Process with MLX Whisper
        result = mlx_whisper.transcribe(
            temp_file_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language=language,
            word_timestamps=word_timestamps,
            fp16=fp16,
            best_of=best_of,
            no_speech_threshold=no_speech_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            condition_on_previous_text=condition_on_previous_text
        )

        return result
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

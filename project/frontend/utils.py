"""
Frontend utility functions for handling transcription data formatting.

This module contains helper functions for formatting timestamps and subtitles
for the MLX Whisper transcription application.
"""
from io import BytesIO


def format_time(seconds: float) -> str:
    """
    Format seconds as MM:SS.mmm for custom timestamp format.

    Args:
        seconds: Time in seconds (float)

    Returns:
        Formatted time string in MM:SS.mmm format
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    # Format to ensure we have 2 digits for minutes, 2 digits for seconds, and 3 digits for milliseconds
    milliseconds = int((remaining_seconds % 1) * 1000)
    whole_seconds = int(remaining_seconds)

    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def format_srt_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS,mmm for SRT format.

    Args:
        seconds: Time in seconds (float)

    Returns:
        Formatted time string in HH:MM:SS,mmm format for SRT subtitles
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    # Format to ensure proper SRT time format: HH:MM:SS,mmm
    milliseconds = int((remaining_seconds % 1) * 1000)
    whole_seconds = int(remaining_seconds)

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def format_timestamps_as_subtitles(result: dict) -> list:
    """
    Convert segments to custom subtitle format with [MM:SS.mmm --> MM:SS.mmm] text.

    Args:
        result: Transcription result dictionary containing segments

    Returns:
        List of formatted subtitle strings
    """
    formatted_lines = []

    if "segments" not in result or not result["segments"]:
        return formatted_lines

    for segment in result["segments"]:
        if "start" not in segment or "end" not in segment or "text" not in segment:
            continue

        start_formatted = format_time(segment["start"])
        end_formatted = format_time(segment["end"])

        line = f"[{start_formatted} --> {end_formatted}]  {segment['text'].strip()}"
        formatted_lines.append(line)

    return formatted_lines


def generate_srt_content(result: dict) -> str:
    """
    Generate SRT (SubRip Text) formatted content.

    Args:
        result: Transcription result dictionary containing segments

    Returns:
        String containing properly formatted SRT subtitle content
    """
    srt_lines = []

    if "segments" not in result or not result["segments"]:
        return ""

    for i, segment in enumerate(result["segments"], 1):
        if "start" not in segment or "end" not in segment or "text" not in segment:
            continue

        # SRT index number
        srt_lines.append(str(i))

        # SRT timestamp line
        start_formatted = format_srt_time(segment["start"])
        end_formatted = format_srt_time(segment["end"])
        srt_lines.append(f"{start_formatted} --> {end_formatted}")

        # SRT text line
        srt_lines.append(segment["text"].strip())

        # Empty line between entries
        srt_lines.append("")

    return "\n".join(srt_lines)


def get_subtitle_download_buttons(result: dict, file_info: dict) -> tuple:
    """
    Generate download buttons for various subtitle formats.

    Args:
        result: Transcription result dictionary
        file_info: Dictionary containing file information including name

    Returns:
        Tuple containing (transcript_bytes, srt_bytes, json_bytes) for download buttons
    """
    # Basic transcript
    transcript_bytes = result["text"].encode("utf-8")

    # SRT format
    srt_content = generate_srt_content(result)
    srt_bytes = srt_content.encode("utf-8") if srt_content else None

    # Custom subtitle format
    formatted_lines = format_timestamps_as_subtitles(result)
    subtitle_bytes = "\n".join(formatted_lines).encode(
        "utf-8") if formatted_lines else None

    # JSON format (full data)
    import json
    json_bytes = json.dumps(result, indent=2).encode("utf-8")

    return transcript_bytes, srt_bytes, subtitle_bytes, json_bytes

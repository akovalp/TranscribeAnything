# frontend/app.py
import streamlit as st
import requests
import time
import json
from io import BytesIO
import logging  # Optional: for better debugging
# Import utility functions for timestamp formatting
from utils import (
    format_time,
    format_srt_time,
    format_timestamps_as_subtitles,
    generate_srt_content
)

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the FastAPI backend URL
API_URL = "http://localhost:8000"  # Ensure this points to your running backend

# Initialize session state variables
if 'transcription_state' not in st.session_state:
    # idle, running, cancelled, success, error
    st.session_state.transcription_state = "idle"
if 'result' not in st.session_state:
    st.session_state.result = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'uploaded_file_info' not in st.session_state:
    st.session_state.uploaded_file_info = None  # To store name/type if needed

# --- Helper Functions ---


def reset_state():
    """Resets the transcription state."""
    st.session_state.transcription_state = "idle"
    st.session_state.result = None
    st.session_state.error_message = None
    # Keep uploaded_file_info unless a new file is uploaded


def request_transcription(uploaded_file_obj, settings):
    """Sends the transcription request to the backend."""
    files = {"file": (uploaded_file_obj.name,
                      uploaded_file_obj, uploaded_file_obj.type)}
    form_data = {k: str(v).lower() if isinstance(v, bool) else str(v)
                 for k, v in settings.items() if v is not None}

    try:
        response = requests.post(
            f"{API_URL}/transcribe/",
            files=files,
            data=form_data,
            timeout=1000
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Request timed out.")
        st.session_state.error_message = "The transcription request timed out. The file might be too long or the server busy."
        st.session_state.transcription_state = "error"
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        error_detail = f"Error: {e}"
        try:
            # Try to get more specific error from response body if available
            error_detail = f"Error {response.status_code}: {response.json().get('detail', response.text)}"
        except Exception:
            pass  # Keep the original error if response parsing fails
        st.session_state.error_message = f"Failed to communicate with the transcription server. {error_detail}"
        st.session_state.transcription_state = "error"
        return None
    except Exception as e:  # Catch other potential errors
        logger.error(f"An unexpected error occurred: {e}")
        st.session_state.error_message = f"An unexpected error occurred: {str(e)}"
        st.session_state.transcription_state = "error"
        return None

# --- UI Rendering ---


# App header (unchanged)
st.markdown('<p class="main-header">MLX Whisper Audio Transcription</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subheader">Convert speech to text with Apple Silicon acceleration</p>',
            unsafe_allow_html=True)

# Create sidebar for configuration
with st.sidebar:
    st.header("Transcription Settings")
    # Disable settings while processing
    is_running = st.session_state.transcription_state == "running"

    model = st.selectbox(
        "Model",
        ["whisper-tiny", "whisper-base", "whisper-small",
            "whisper-medium", "whisper-large-v3"],
        index=4,
        help="Select the Whisper model to use for transcription",
        disabled=is_running
    )
    language = st.selectbox(
        "Language",
        [None, "en", "fr", "de", "es", "it", "pt",
            "nl", "ru", "zh", "ja", "ko", "ar"],
        index=0,
        format_func=lambda x: "Auto-detect" if x is None else x,
        help="Select the language of the audio (optional)",
        disabled=is_running
    )

    with st.expander("Advanced Options"):
        word_timestamps = st.toggle(
            "Word Timestamps", value=True, help="Generate timestamps for each word",
            disabled=is_running
        )
        fp16 = st.toggle("Use FP16", value=True,
                         help="Use half-precision floating point (faster)",
                         disabled=is_running)
        best_of = st.slider("Best Of", min_value=1, max_value=50,
                            value=5, help="Number of candidates to generate",
                            disabled=is_running)
        no_speech_threshold = st.slider(
            "No Speech Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01,
            help="Threshold for classifying audio as speech",
            disabled=is_running
        )
        # Simplified Hallucination filter toggle
        enable_hallucination_filter = st.toggle(
            "Enable Hallucination Filter", value=True, disabled=is_running
        )
        hallucination_threshold = st.slider(
            "Hallucination Silence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
            help="Threshold for filtering out hallucinations (active if filter is enabled)",
            disabled=is_running or not enable_hallucination_filter
        )
        # Use the slider value only if the toggle is on
        effective_hallucination_threshold = hallucination_threshold if enable_hallucination_filter else None

        condition_on_previous_text = st.toggle(
            "Condition on Previous Text", value=True, disabled=is_running
        )

    # Store current settings in a dictionary
    transcription_settings = {
        "language": language,
        "word_timestamps": word_timestamps,
        "fp16": fp16,
        "best_of": best_of,
        "no_speech_threshold": no_speech_threshold,
        "hallucination_silence_threshold": effective_hallucination_threshold,
        "condition_on_previous_text": condition_on_previous_text
    }

# Main area
st.subheader("Upload Audio File")
help_text = "Supported formats: mp3, wav, m4a, flac (max 100MB)"
st.caption(help_text)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a", "flac", "mov"],
    disabled=is_running,
    on_change=reset_state  # Reset if a new file is uploaded
)

# --- Transcription Logic and Status Display ---

if uploaded_file is not None:
    # Store file info if not already stored or if it changed
    if st.session_state.uploaded_file_info is None or \
       st.session_state.uploaded_file_info['name'] != uploaded_file.name or \
       st.session_state.uploaded_file_info['size'] != uploaded_file.size:
        st.session_state.uploaded_file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type
        }
        # Reset state if a truly new file is uploaded
        reset_state()

    # Display audio player
    st.audio(uploaded_file, format=uploaded_file.type)

    # Control buttons and status area
    col1, col2 = st.columns([1, 3])

    with col1:
        # Show Transcribe button only when idle and file is present
        if st.session_state.transcription_state == "idle":
            if st.button("Transcribe", type="primary", use_container_width=True):
                st.session_state.transcription_state = "running"
                st.session_state.result = None  # Clear previous results
                st.session_state.error_message = None
                st.rerun()  # Rerun to show the status indicator

    # Display status or results area
    status_placeholder = st.empty()
    result_placeholder = st.container()

    if st.session_state.transcription_state == "running":
        with status_placeholder:
            col_status, col_cancel = st.columns([3, 1])
            with col_status:
                # Using st.status for better visual feedback during processing
                with st.spinner("Transcribing audio... Please wait."):
                    # This block now mainly waits. The actual request happens below.
                    # We don't use a fake progress bar anymore.
                    # You can update this message if backend provided stages
                    st.write("Processing...")

            with col_cancel:
                if st.button("Cancel", type="secondary", use_container_width=True):
                    st.session_state.transcription_state = "cancelled"
                    # NOTE: This doesn't stop the backend process, only the frontend waiting.
                    logger.info("Transcription cancelled by user.")
                    st.rerun()  # Rerun to update UI immediately

        # Perform the actual transcription request outside the spinner for clarity
        # This check ensures we only run the request once per "running" state trigger
        # We rely on rerun triggering this part *after* state is set to running
        if st.session_state.result is None and st.session_state.error_message is None:
            # Make the blocking request here
            api_result = request_transcription(
                uploaded_file, transcription_settings)

            # Check if cancelled *during* the request
            if st.session_state.transcription_state == "cancelled":
                logger.info("Processing finished after cancellation request.")
                # Don't proceed to success/error state if cancelled
                st.rerun()

            elif api_result:
                st.session_state.result = api_result
                st.session_state.transcription_state = "success"
                st.rerun()  # Rerun to display results
            # If request_transcription didn't already set error
            elif st.session_state.transcription_state != "error":
                st.session_state.error_message = "Transcription failed for an unknown reason."
                st.session_state.transcription_state = "error"
                st.rerun()  # Rerun to display error

    elif st.session_state.transcription_state == "cancelled":
        status_placeholder.warning("Transcription cancelled by user.")
        # Optionally add a button to clear the cancel state and allow retrying
        if st.button("Clear Status", use_container_width=True):
            reset_state()
            st.rerun()

    elif st.session_state.transcription_state == "success" and st.session_state.result:
        status_placeholder.success("Transcription complete!")
        result = st.session_state.result
        file_info = st.session_state.uploaded_file_info

        with result_placeholder:
            st.subheader("Transcription Text")
            st.markdown(
                f'<div class="result-area">{result["text"]}</div>', unsafe_allow_html=True)

            detected_language = result.get("language", "unknown")
            st.info(f"Detected language: {detected_language}")

            # Add formatted timestamps displays
            if "segments" in result and result["segments"]:
                # Create tabs for different timestamp formats
                tab1, tab2, tab3 = st.tabs(
                    ["Subtitle Format", "SRT Format", "Word Timestamps"])

                with tab1:
                    # Custom subtitle format
                    formatted_lines = format_timestamps_as_subtitles(result)
                    if formatted_lines:
                        formatted_text = "\n".join(formatted_lines)
                        st.text_area("Subtitle Format",
                                     formatted_text, height=300)

                        subtitle_bytes = formatted_text.encode("utf-8")
                        st.download_button(
                            label="Download Formatted Timestamps",
                            data=BytesIO(subtitle_bytes),
                            file_name=f"{file_info['name'].split('.')[0]}_subtitles.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "No segment data available to format timestamps.")

                with tab2:
                    # SRT format
                    srt_content = generate_srt_content(result)
                    if srt_content:
                        st.text_area("SRT Format", srt_content, height=300)

                        srt_bytes = srt_content.encode("utf-8")
                        st.download_button(
                            label="Download SRT Subtitle File",
                            data=BytesIO(srt_bytes),
                            file_name=f"{file_info['name'].split('.')[0]}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "No segment data available to generate SRT file.")

                with tab3:
                    # Original word timestamps (only show if word_timestamps was requested)
                    word_timestamps_requested = transcription_settings.get(
                        "word_timestamps", False)
                    if word_timestamps_requested and "segments" in result and result["segments"]:
                        # Check if first segment has 'words' key
                        if result["segments"][0] and "words" in result["segments"][0]:
                            segments = result["segments"]
                            for i, segment in enumerate(segments):
                                st.markdown(
                                    f"**Segment {i+1}:** {segment.get('text', '')}")
                                if "words" in segment:
                                    words_data = []
                                    for word in segment["words"]:
                                        words_data.append({
                                            "Word": word.get("word", ""),
                                            "Start": f"{word.get('start', 0):.2f}s",
                                            "End": f"{word.get('end', 0):.2f}s"
                                        })
                                    st.dataframe(words_data)
                        else:
                            st.caption(
                                "Word timestamps were requested but not generated by the model for this audio.")
                    else:
                        st.info(
                            "Word-level timestamps were not requested in the transcription settings.")

            # Download Buttons section - expanded with columns for all download options
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                transcript_bytes = result["text"].encode("utf-8")
                st.download_button(
                    label="Download Transcript (.txt)",
                    data=BytesIO(transcript_bytes),
                    file_name=f"{file_info['name'].split('.')[0]}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # Add SRT download in the second column
            if "segments" in result and result["segments"]:
                with col_dl2:
                    srt_content = generate_srt_content(result)
                    if srt_content:
                        srt_bytes = srt_content.encode("utf-8")
                        st.download_button(
                            label="Download SRT Subtitles",
                            data=BytesIO(srt_bytes),
                            file_name=f"{file_info['name'].split('.')[0]}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )

            # Keep JSON download in the third column
            word_timestamps_requested = transcription_settings.get(
                "word_timestamps", False)
            if word_timestamps_requested and "segments" in result and result["segments"] and result["segments"][0] and "words" in result["segments"][0]:
                with col_dl3:
                    json_bytes = json.dumps(result, indent=2).encode("utf-8")
                    st.download_button(
                        label="Download Full JSON",
                        data=BytesIO(json_bytes),
                        file_name=f"{file_info['name'].split('.')[0]}_full_transcript.json",
                        mime="application/json",
                        use_container_width=True
                    )

    elif st.session_state.transcription_state == "error":
        if st.session_state.error_message:
            status_placeholder.error(
                f"An error occurred: {st.session_state.error_message}")
        else:
            status_placeholder.error(
                "An unknown error occurred during transcription.")
        # Optionally add a button to clear the error state and allow retrying
        if st.button("Clear Error", use_container_width=True):
            reset_state()
            st.rerun()


# Footer (unchanged)
st.markdown("---")
st.caption("Powered by MLX Whisper for Apple Silicon devices")

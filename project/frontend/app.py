import streamlit as st
import requests
import time
import json
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="MLX Whisper Transcription",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4D5D8D;
        margin-bottom: 2rem;
    }
    .result-area {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Define the FastAPI backend URL
API_URL = "http://localhost:8000"

# App header
st.markdown('<p class="main-header">MLX Whisper Audio Transcription</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subheader">Convert speech to text with Apple Silicon acceleration</p>',
            unsafe_allow_html=True)

# Create sidebar for configuration
with st.sidebar:
    st.header("Transcription Settings")

    # Model selection (if you have multiple models)
    model = st.selectbox(
        "Model",
        ["whisper-tiny", "whisper-base", "whisper-small",
            "whisper-medium", "whisper-large-v3"],
        index=4,
        help="Select the Whisper model to use for transcription"
    )

    # Language selection
    language = st.selectbox(
        "Language",
        [None, "en", "fr", "de", "es", "it", "pt",
            "nl", "ru", "zh", "ja", "ko", "ar"],
        index=0,
        format_func=lambda x: "Auto-detect" if x is None else x,
        help="Select the language of the audio (optional)"
    )

    # Advanced options - in an expander to save space
    with st.expander("Advanced Options"):
        word_timestamps = st.toggle(
            "Word Timestamps", value=True, help="Generate timestamps for each word")
        fp16 = st.toggle("Use FP16", value=True,
                         help="Use half-precision floating point (faster)")
        best_of = st.slider("Best Of", min_value=1, max_value=50,
                            value=5, help="Number of candidates to generate")
        no_speech_threshold = st.slider(
            "No Speech Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            help="Threshold for classifying audio as speech"
        )
        hallucination_threshold = st.slider(
            "Hallucination Silence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1 if st.toggle(
                "Enable Hallucination Filter", value=True) else None,
            step=0.01,
            help="Threshold for filtering out hallucinations"
        )
        condition_on_previous_text = st.toggle(
            "Condition on Previous Text", value=True)

# Main area for file upload and results
st.subheader("Upload Audio File")
help_text = """
Supported formats: mp3, wav, m4a, flac (max 100MB)
"""
st.caption(help_text)

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=[
                                 "mp3", "wav", "m4a", "flac"])

# Handle file upload and transcription
if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[1]}")

    # Create columns for buttons
    col1, col2 = st.columns([1, 3])

    with col1:
        # Transcribe button
        if st.button("Transcribe", type="primary", use_container_width=True):
            # Create multipart form data
            files = {"file": (uploaded_file.name, uploaded_file,
                              f"audio/{uploaded_file.type.split('/')[1]}")}

            # Prepare the form data
            form_data = {
                "language": language if language else None,
                "word_timestamps": str(word_timestamps).lower(),
                "fp16": str(fp16).lower(),
                "best_of": str(best_of),
                "no_speech_threshold": str(no_speech_threshold),
                "hallucination_silence_threshold": str(hallucination_threshold) if hallucination_threshold is not None else None,
                "condition_on_previous_text": str(condition_on_previous_text).lower()
            }

            # Remove None values
            form_data = {k: v for k, v in form_data.items() if v is not None}

            # Start spinner and progress for better UX
            with st.spinner("Transcribing audio... Please wait"):
                try:
                    # For better UX, let's use a progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Update progress bar (simulated progress)
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)

                    # Make the request to FastAPI backend
                    response = requests.post(
                        f"{API_URL}/transcribe/",
                        files=files,
                        data=form_data,
                        timeout=300  # 5 minute timeout for large files
                    )

                    # Check if successful
                    if response.status_code == 200:
                        # Parse the response
                        result = response.json()

                        # Display the results
                        st.success("Transcription complete!")

                        # Show the transcription
                        st.subheader("Transcription Text")
                        st.markdown(
                            f'<div class="result-area">{result["text"]}</div>', unsafe_allow_html=True)

                        # Show detected language
                        detected_language = result.get("language", "unknown")
                        st.info(f"Detected language: {detected_language}")

                        # If word timestamps were requested, show them
                        if word_timestamps and "segments" in result:
                            with st.expander("View Word Timestamps"):
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

                        # Add a download button for the transcript
                        transcript_bytes = result["text"].encode("utf-8")
                        st.download_button(
                            label="Download Transcript",
                            data=BytesIO(transcript_bytes),
                            file_name=f"{uploaded_file.name.split('.')[0]}_transcript.txt",
                            mime="text/plain",
                        )

                        # If word timestamps were requested, provide option to download full JSON
                        if word_timestamps and "segments" in result:
                            json_bytes = json.dumps(
                                result, indent=2).encode("utf-8")
                            st.download_button(
                                label="Download Full JSON",
                                data=BytesIO(json_bytes),
                                file_name=f"{uploaded_file.name.split('.')[0]}_full_transcript.json",
                                mime="application/json",
                            )

                    else:
                        st.error(
                            f"Error: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with col2:
        # Optionally add another button or feature here
        pass

# Footer with information
st.markdown("---")
st.caption("Powered by MLX Whisper for Apple Silicon devices")

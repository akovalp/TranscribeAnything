
# MLX Whisper Transcription API & Frontend


![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-lightgrey?logo=apple)


A full-stack application demonstrating speech-to-text transcription using Apple's MLX framework with the Whisper model, optimized for Apple Silicon (M1/M2/M3 chips). This project features a FastAPI backend API and an interactive Streamlit web frontend.


---

## 🌟 Features

*   **FastAPI Backend:** Provides a robust API endpoint (`/transcribe/`) for audio transcription requests.
*   **Streamlit Frontend:** User-friendly interface to upload audio files, configure transcription settings, view progress, and see results.
*   **MLX Whisper Integration:** Leverages Apple's MLX framework for efficient Whisper model execution on Apple Silicon hardware (using `mlx-community/whisper-large-v3-mlx` by default).
*   **Configurable Transcription:** Adjust settings like language (or auto-detect), word timestamps, precision (FP16), and advanced Whisper parameters.
*   **Multiple Audio Formats:** Supports `.mp3`, `.wav`, `.m4a`, `.flac`, `.mov`.
*   **Result Display:** Shows the full transcript, detected language, and formatted segments.
*   **Download Options:** Download the transcript as a plain text file (`.txt`), SRT subtitle file (`.srt`), or the full JSON response with detailed timestamps.
*   **Word-Level Timestamps:** Option to generate and view timestamps for individual words.
*   **Status & Error Handling:** Provides feedback during transcription and clear error messages.

---

## 📸 Example Flow

https://github.com/user-attachments/assets/353ad8ce-9569-4d25-a92d-6b8fa539bc41




---

## 💻 Technology Stack

*   **Backend:** Python, FastAPI, Uvicorn, MLX, mlx-whisper
*   **Frontend:** Python, Streamlit
*   **Testing:** Python, Requests
*   **Audio Processing Dependency:** FFmpeg

---

## 📂 Project Structure

```
.
├── backend/
│   ├── main.py         # FastAPI application logic
│   ├── models.py       # Pydantic models for request/response
│   └── utils.py        # Transcription utility functions (using mlx-whisper)
├── frontend/
│   ├── app.py          # Streamlit application logic
│   └── utils.py        # Frontend utility functions (formatting timestamps)
├── test.py             # Simple API test script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

*   **Hardware:** A Mac with Apple Silicon (M1, M2, M3, etc.). MLX is specifically designed for this hardware.
*   **Operating System:** macOS
*   **Python:** Version 3.8 or newer. Check with `python3 --version`.
*   **Pip:** Python package installer. Usually comes with Python. Check with `pip3 --version`.
*   **Git:** Version control system. Check with `git --version`. Install from [git-scm.com](https://git-scm.com/) if needed.
*   **FFmpeg:** A command-line tool for handling audio/video. Whisper (and `mlx-whisper`) needs this to process various audio formats.
    *   **Installation (Recommended):** Use Homebrew (the macOS package manager). If you don't have Homebrew, install it from [brew.sh](https://brew.sh/). Then, open Terminal and run:
        ```bash
        brew install ffmpeg
        ```

### Installation Steps

1.  **Clone the Repository:**
    Open your Terminal application and run the following command (replace `<https://github.com/your-user-nname/TranscribeAnything>` with the actual URL):
    ```bash
    git clone <your-repository-url>
    cd mlx-whisper-transcription-app # Or your repository's folder name
    ```
    Please dont forget to change `your-user-name` to your actual GitHub username.

2.  **Create a Virtual Environment:**
    It's best practice to keep project dependencies separate.
    ```bash
    python3 -m venv venv
    ```
    This creates a folder named `venv` in your project directory.

3.  **Activate the Virtual Environment:**
    You need to activate the environment *each time* you work on the project in a new terminal session.
    ```bash
    source venv/bin/activate
    ```
    Your terminal prompt should now show `(venv)` at the beginning.

4.  **Install Dependencies:**
    Make sure you have a `requirements.txt` file in the main project folder (see section below if you need to create it). Install all necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This might take a few minutes, especially downloading the ML/MLX packages.


---

## ▶️ Running the Application

You need to run the backend API and the frontend application separately, usually in two different Terminal windows/tabs. **Make sure your virtual environment is activated (`source venv/bin/activate`) in both terminals.**

1.  **Start the Backend (FastAPI API):**
    *   Navigate to the `backend` directory:
        ```bash
        cd backend
        ```
    *   Start the Uvicorn server:
        ```bash
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```
        *   `main:app`: Tells Uvicorn to find the FastAPI `app` object inside the `main.py` file.
        *   `--reload`: Automatically restarts the server when you save changes to the code (useful for development).
        *   `--host 0.0.0.0`: Makes the server accessible from other devices on your network (and needed for Streamlit to connect).
        *   `--port 8000`: Runs the server on port 8000.
    *   You should see output indicating the server is running, like `Uvicorn running on http://0.0.0.0:8000`. Keep this terminal open!

2.  **Start the Frontend (Streamlit App):**
    *   Open a **new** Terminal window or tab.
    *   **Activate the virtual environment** again in this new terminal:
        ```bash
        # Navigate back to the root project directory first if needed
        # cd ..
        source venv/bin/activate
        ```
    *   Navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    *   Streamlit should automatically open a new tab in your web browser pointing to the application (usually `http://localhost:8501`). If not, the terminal output will provide the URL.
    *   Keep this terminal open too!

3.  **Interact:** Use the Streamlit interface in your browser to upload audio files and get transcriptions. The frontend will communicate with the backend API running on port 8000.

---

## 🧪 Testing the API

A simple test script (`test.py`) is included to directly test the backend API endpoint without using the frontend.

1.  Make sure the backend server is running (Step 1 in "Running the Application").
2.  Open a **new** Terminal window/tab (or use one where the backend/frontend aren't running).
3.  **Activate the virtual environment:** `source venv/bin/activate`.
4.  Navigate to the **root** project directory (the one containing `test.py`).
5.  Run the script, providing the path to an audio file you want to test:
    ```bash
    python test.py /path/to/your/audio.mp3
    ```
    *(Replace `/path/to/your/audio.mp3` with an actual audio file path)*
6.  The script will send the file to the API and print the JSON response (or any errors) to the terminal.

---

## 🔮 Future Improvements / Ideas

*   **Asynchronous Processing:** For very long audio files, move the `mlx-whisper` transcription to a background task (e.g., using FastAPI's `BackgroundTasks` or Celery) so the API request returns immediately, and the frontend polls for results.
*   **Dockerization:** Containerize the application (Note: Docker on macOS with MLX acceleration can be complex).
*   **Error Handling:** More granular error handling and user feedback on both frontend and backend.
*   **Progress Indication:** Provide more detailed progress updates from the backend to the frontend during transcription (if feasible with `mlx-whisper`).
*   **Deployment:** Explore options for deploying (keeping in mind the Apple Silicon dependency might limit cloud options or require specific hardware).
*   **UI Enhancements:** Improve the Streamlit UI/UX further.

---



## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details (or choose another license if you prefer).

---

## 📧 Contact

Alp Akova – akovalp@gmail

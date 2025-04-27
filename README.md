# Voice Coach

A simple real-time voice analysis tool using Vosk, Parselmouth, Librosa, and Electron to provide feedback on speaking habits like pitch (F0), volume (RMS), upward inflection, staccato rhythm, and vocal shakiness (jitter/shimmer).

This is a pet project primarily developed for macOS.

## Features

*   Real-time calculation of RMS (volume) and F0 (pitch).
*   Utterance-level analysis of:
    *   Jitter & Shimmer (using Praat via Parselmouth)
    *   Staccato Rhythm (based on pause/word duration statistics from Vosk)
    *   Upward Inflection (based on F0 slope at utterance end)
*   Simple Electron UI displaying metrics and visual alerts.

## Tech Stack

*   **Backend:** Python 3
*   **Speech-to-Text:** Vosk
*   **Acoustic Analysis:** Parselmouth (Praat), Librosa
*   **Audio I/O:** sounddevice
*   **Frontend:** Electron, Node.js

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd voicecoach
    ```

2.  **Python Setup (Requires Python 3.9+):**
    *   Create a virtual environment:
        ```bash
        python3 -m venv venv
        ```
    *   Activate the environment:
        *   macOS/Linux: `source venv/bin/activate`
        *   Windows: `.\venv\Scripts\activate`
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Download Vosk Model:**
    *   Download the model `vosk-model-small-en-us-0.15` from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).
    *   Extract the downloaded archive.
    *   **IMPORTANT:** Place the extracted folder (which should be named `vosk-model-small-en-us-0.15`) directly into the root directory of this project (`voicecoach/`).

4.  **Node.js Setup (Requires Node >= v22, npm >= v10):**
    *   Install Node.js and npm if you haven't already: [https://nodejs.org/](https://nodejs.org/)
    *   Install Node dependencies:
        ```bash
        npm install
        ```

## Running the App

Ensure your Python virtual environment is **deactivated**. From the project's root directory (`voicecoach/`), run:

```bash
npm start
```

The application window should open and automatically start listening.

## Key Files

*   `voice.py`: Python backend script handling audio capture, analysis, and JSON output.
*   `main.js`: Electron main process script, manages the app window and Python child process.
*   `preload.js`: Electron preload script for secure IPC.
*   `renderer.js`: Electron renderer process script, handles UI logic and updates.
*   `index.html`: Defines the UI structure.

## Known Issues/Limitations

*   Primarily tested on macOS.
*   Shakiness detection uses basic Jitter/Shimmer thresholds that may need tuning (`renderer.js`).
*   Staccato detection rules are experimental (`voice.py`).
*   Requires the specific Vosk model `vosk-model-small-en-us-0.15` placed in the root folder. 
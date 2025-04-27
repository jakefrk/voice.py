import sounddevice as sd
import numpy as np
import sys
import time
import json # To parse Vosk results
import os
import librosa # Import librosa

# --- Vosk Configuration ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15" # CHANGE if you downloaded a different model
# --------------------------

# --- Audio Configuration ---
# SAMPLE_RATE = 16000 # Vosk models usually prefer 16kHz or 8kHz
# Let's stick with the device default for now and see if Vosk handles it
# CHUNK_SIZE = 8000 # Process audio in chunks
DEVICE = None        # Use default input device
CHANNELS = 1         # Mono audio
DTYPE = 'int16'      # Data type (Vosk prefers int16)
# --- Librosa Configuration ---
# We need the sample rate BEFORE the callback, so get it during init
SAMPLE_RATE = None # Will be determined later
# --- Librosa F0 Configuration ---
FMIN = librosa.note_to_hz('C2') # Min pitch freq (~65 Hz)
FMAX = librosa.note_to_hz('C7') # Max pitch freq (~2093 Hz)
# --------------------------

# --- Vosk Initialization ---
try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    print("Please install the 'vosk' library: pip install vosk", file=sys.stderr)
    sys.exit(1)

if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model folder not found at '{VOSK_MODEL_PATH}'.", file=sys.stderr)
    print("Please download a model from https://alphacephei.com/vosk/models", file=sys.stderr)
    print("Unzip it and place the folder in your project directory.", file=sys.stderr)
    sys.exit(1)

print(f"Loading Vosk model from: {VOSK_MODEL_PATH}")
model = Model(VOSK_MODEL_PATH)
print("Vosk model loaded successfully.")
# --------------------------

# --- Get Device Info, Initialize Recognizer & Set Sample Rate ---
try:
    default_input_info = sd.query_devices(DEVICE, 'input')
    # Set the global SAMPLE_RATE variable
    SAMPLE_RATE = int(default_input_info['default_samplerate'])
    print(f"Using default input device: {default_input_info['name']}")
    print(f"Using sample rate: {SAMPLE_RATE} Hz")

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    print("Vosk recognizer initialized.")

    # CHUNK_SIZE based on sample rate (e.g., 0.5 seconds)
    CHUNK_SIZE = int(SAMPLE_RATE * 0.5)
    print(f"Using chunk size: {CHUNK_SIZE} samples")

except Exception as e:
    print(f"Error during initialization: {e}", file=sys.stderr)
    sys.exit(1)
# ------------------------------------

# --- Librosa RMS Calculation Helper ---
# Calculate the maximum value for int16 for normalization
INT16_MAX = np.iinfo(np.int16).max

def process_audio_chunk(audio_chunk_int16):
    """Converts int16 chunk to float32, normalized."""
    # Reshape needed if audio_chunk_int16 has shape (N, 1) -> (N,)
    return audio_chunk_int16.astype(np.float32).flatten() / INT16_MAX

def calculate_rms(audio_chunk_float32):
    """Calculates RMS energy of a float32 audio chunk."""
    rms = librosa.feature.rms(y=audio_chunk_float32)
    avg_rms = np.mean(rms)
    return avg_rms

def calculate_f0(audio_chunk_float32, sr):
    """Calculates fundamental frequency (F0/pitch) of a float32 audio chunk using YIN."""
    # Use librosa.yin instead of librosa.pyin
    f0 = librosa.yin(
        y=audio_chunk_float32,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr
    )
    # librosa.yin returns F0 per frame. Find the average where F0 > 0 (is voiced)
    # It might return NaN for unvoiced frames, so use nanmean
    avg_f0 = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
    return avg_f0
# ------------------------------------


def audio_callback(indata, frames, time_info, status):
    """This function is called for each audio chunk."""
    global recognizer, SAMPLE_RATE # Need SAMPLE_RATE here

    if status:
        print(status, file=sys.stderr)
        return

    # 1. Convert chunk to float32 once
    audio_float32 = process_audio_chunk(indata)

    # 2. Process with Vosk
    vosk_processed = False
    if recognizer.AcceptWaveform(indata.tobytes()):
        result_json = recognizer.Result()
        result_dict = json.loads(result_json)
        if 'text' in result_dict and result_dict['text']:
             print(f"\nVOSK FINAL: {result_dict['text']}") # Add newline for clarity
        vosk_processed = True # Mark that Vosk gave a final result this chunk
    else:
        partial_json = recognizer.PartialResult()
        partial_dict = json.loads(partial_json)
        if 'partial' in partial_dict and partial_dict['partial']:
            # Clear the line before printing partial result
            print(" " * 90, end='\r') # Clear previous partial result
            print(f"VOSK Partial: {partial_dict['partial']}", end='\r', flush=True)

    # 3. Process with Librosa
    current_rms = calculate_rms(audio_float32)
    current_f0 = calculate_f0(audio_float32, SAMPLE_RATE)

    # 4. Print Features
    # Only update the feature line if Vosk didn't just print a final result
    if not vosk_processed:
        print(f"RMS: {current_rms:.4f} | F0: {current_f0:7.2f} Hz      ", end='\r', flush=True)
    else:
        # If Vosk printed final result, print features on a new line and clear it after
        print(f"RMS: {current_rms:.4f} | F0: {current_f0:7.2f} Hz      ")
        # Optional: Clear the feature line immediately after final Vosk output
        # print(" " * 90, end='\r', flush=True)


try:
    print("\n--- Starting Real-Time Transcription, RMS & F0 Analysis ---")
    print("Speak into your microphone. Press Ctrl+C to stop.")

    with sd.InputStream(
        device=DEVICE,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype=DTYPE,
        callback=audio_callback
    ):
        while True:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping analysis.")
except Exception as e:
    print(f"An error occurred during audio streaming: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    # Final Vosk result
    if 'recognizer' in locals() and hasattr(recognizer, 'FinalResult'):
        final_result_json = recognizer.FinalResult()
        try: # Add try-except for JSON parsing robustness
            final_result_dict = json.loads(final_result_json)
            if 'text' in final_result_dict and final_result_dict['text']:
                 print(f"\nVOSK FINAL (on exit): {final_result_dict['text']}")
        except json.JSONDecodeError:
            print("\nCould not decode final Vosk result.")
    print("Script finished.")
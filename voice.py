import sounddevice as sd
import numpy as np
import sys
import time
import json
import os
import librosa
import parselmouth
from collections import deque

# --- Vosk Configuration ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
# --------------------------

# --- Audio Configuration ---
DEVICE = None
CHANNELS = 1
DTYPE = 'int16'
SAMPLE_RATE = None
# --------------------------

# --- Librosa F0 Configuration ---
FMIN = librosa.note_to_hz('C2')
FMAX = librosa.note_to_hz('C7')
# -----------------------------

# --- Parselmouth Configuration ---
# We still need these for pitch calculation within parselmouth
# PRAAT_FMIN = 75.0 # Not strictly needed now if using defaults in to_pitch_ac
# PRAAT_FMAX = 600.0 # Not strictly needed now if using defaults in to_pitch_ac
# -----------------------------

# --- Analysis Buffer ---
BUFFER_DURATION_SECONDS = 3 # Analyze jitter/shimmer over the last N seconds on utterance end
MAX_BUFFER_CHUNKS = None
audio_buffer = None
# ----------------------


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

# --- Init Recognizer, Get Sample Rate & Chunk Size, Init Buffer ---
try:
    default_input_info = sd.query_devices(DEVICE, 'input')
    SAMPLE_RATE = int(default_input_info['default_samplerate'])
    print(f"Using default input device: {default_input_info['name']}")
    print(f"Using sample rate: {SAMPLE_RATE} Hz")

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    print("Vosk recognizer initialized.")

    CHUNK_DURATION = 0.5
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    print(f"Using chunk size: {CHUNK_SIZE} samples ({CHUNK_DURATION}s)")

    MAX_BUFFER_CHUNKS = int(BUFFER_DURATION_SECONDS / CHUNK_DURATION)
    audio_buffer = deque(maxlen=MAX_BUFFER_CHUNKS)
    print(f"Audio buffer size: {MAX_BUFFER_CHUNKS} chunks ({BUFFER_DURATION_SECONDS}s)")


except Exception as e:
    print(f"Error during initialization: {e}", file=sys.stderr)
    sys.exit(1)
# ------------------------------------

# --- Helper Functions ---
INT16_MAX = np.iinfo(np.int16).max

def process_audio_chunk_float(audio_chunk_int16):
    """Converts int16 chunk to float32, normalized."""
    return audio_chunk_int16.astype(np.float32).flatten() / INT16_MAX

def calculate_rms(audio_chunk_float32):
    """Calculates RMS energy."""
    rms = librosa.feature.rms(y=audio_chunk_float32)
    avg_rms = np.mean(rms)
    return avg_rms

def calculate_f0_librosa(audio_chunk_float32, sr):
    """Calculates F0 using Librosa YIN."""
    f0 = librosa.yin(y=audio_chunk_float32, fmin=FMIN, fmax=FMAX, sr=sr)
    avg_f0 = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0
    return avg_f0

def calculate_jitter_shimmer(long_audio_chunk_float32, sr):
    """Calculates jitter (RAP) and shimmer (local) using Parselmouth."""
    try:
        snd = parselmouth.Sound(long_audio_chunk_float32.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac()
        voiced_frames = pitch.count_voiced_frames()
        if voiced_frames < 10: return 0.0, 0.0

        point_process = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 2: return 0.0, 0.0

        jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100

        if np.isnan(jitter_rap): jitter_rap = 0.0
        if np.isnan(shimmer_local): shimmer_local = 0.0
        return jitter_rap, shimmer_local
    except Exception as e:
        print(f"Parselmouth calculation error: {e}", file=sys.stderr)
        return 0.0, 0.0

def analyze_timing(result_dict):
    """Analyzes word timings for staccato patterns."""
    words = result_dict.get('result', [])
    if not words: return # No timing analysis if no words

    pauses, durs = [], []
    last_end = words[0]['start'] # Initialize with start of first word for potential initial pause calculation if needed
    for i, w in enumerate(words):
        # Ensure timings are valid floats
        start_time = float(w.get('start', -1))
        end_time = float(w.get('end', -1))
        if start_time < 0 or end_time < 0: continue

        # Calculate word duration
        d = end_time - start_time
        if d < 0: d = 0 # Duration cannot be negative
        durs.append(d)

        # Calculate pause before this word (if not the first word)
        if i > 0:
             p = start_time - last_end
             if p < 0: p = 0.0 # Pause cannot be negative
             pauses.append(p)

        last_end = end_time # Update for next pause calculation


    # Compute stats
    avg_p, max_p, std_p = np.mean(pauses) if pauses else 0.0, np.max(pauses) if pauses else 0.0, np.std(pauses) if pauses else 0.0
    avg_d, std_d = np.mean(durs) if durs else 0.0, np.std(durs) if durs else 0.0
    total_t = last_end - float(words[0].get('start', 0.0)) # Use float for start time
    rate = len(words) / total_t if total_t > 0.1 else 0.0

    # Print cleaned-up stats
    print(f"  [Timing] AvgPause={avg_p:.3f}s | MaxPause={max_p:.3f}s | StdPause={std_p:.3f}s")
    print(f"  [Timing] AvgWordDur={avg_d:.3f}s | StdWordDur={std_d:.3f}s | Rate={rate:.2f} wps")

    # Refined staccato flag logic
    flags = []
    if pauses and max_p > 1.0 and max_p > avg_p * 3 and avg_p > 0: flags.append(f"LongPause({max_p:.2f}s)")
    if pauses and avg_p > 0 and std_p > avg_p * 0.8: flags.append(f"VarPauses({std_p:.2f}s)")
    if rate > 4.0: flags.append(f"FastRate({rate:.1f}wps)")
    # Removed SlowRate flag
    # Removed VariableWordDur flag (can be added back if needed)

    if len(flags) >= 2: # Require 2 flags
        print(f"  [Staccato Alert] {', '.join(flags)}")

# ------------------------------------

def audio_callback(indata, frames, time_info, status):
    """This function is called for each audio chunk."""
    global recognizer, SAMPLE_RATE, audio_buffer

    if status:
        print(status, file=sys.stderr)
        return

    # 1. Append raw int16 data to buffer
    audio_buffer.append(indata.copy())

    # 2. Convert current chunk to float32 for immediate RMS/F0
    audio_float32 = process_audio_chunk_float(indata)
    current_rms = calculate_rms(audio_float32)
    current_f0 = calculate_f0_librosa(audio_float32, SAMPLE_RATE)

    # 3. Print immediate features (RMS, F0)
    print(f"RMS: {current_rms:.4f} | F0: {current_f0:7.2f} Hz          ", end='\r', flush=True)

    # 4. Process with Vosk
    vosk_processed = False
    if recognizer.AcceptWaveform(indata.tobytes()):
        vosk_processed = True
        result_json = recognizer.Result()
        try:
            result_dict = json.loads(result_json)
        except json.JSONDecodeError:
            print("\n[Error] Failed to decode Vosk JSON result.", file=sys.stderr)
            result_dict = {}

        final_text = result_dict.get('text', '')

        if final_text:
            print("\n" + "="*40)
            print(f"VOSK FINAL: {final_text}")

            # --- Analyze Jitter/Shimmer on Buffered Audio ---
            if len(audio_buffer) > 1:
                buffered_int16_data = np.concatenate(list(audio_buffer), axis=0)
                buffered_float32_data = process_audio_chunk_float(buffered_int16_data)
                print(f"Analyzing {len(buffered_int16_data)/SAMPLE_RATE:.2f}s of buffered audio...")
                segment_jitter, segment_shimmer = calculate_jitter_shimmer(buffered_float32_data, SAMPLE_RATE)
                print(f"Segment Jitter (RAP): {segment_jitter:5.2f}% | Segment Shimmer: {segment_shimmer:5.2f}%")

            # --- Analyze Word Timing for Staccato ---
            if 'result' in result_dict:
                analyze_timing(result_dict)
            else:
                 print("  [Timing] Word timing information ('result' key) not available in Vosk result.")

            print("="*40 + "\n")
            print(" " * 60, end='\r', flush=True) # Clear line after analysis


    # Handle partial results separately
    if not vosk_processed:
        partial_json = recognizer.PartialResult()
        partial_dict = json.loads(partial_json)
        if 'partial' in partial_dict and partial_dict['partial']:
            print(f"VOSK Partial: {partial_dict['partial']}{' ' * 20}", end='\r', flush=True)


# --- Main Loop ---
try:
    print("\n--- Starting Real-Time Analysis (RMS, F0 | Jitter, Shimmer, Timing on Utterance End) ---")
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
    # Final Vosk result on exit
    if 'recognizer' in locals() and hasattr(recognizer, 'FinalResult'):
        final_result_json = recognizer.FinalResult()
        try:
            final_result_dict = json.loads(final_result_json)
            if 'text' in final_result_dict and final_result_dict['text']:
                 print(f"\nVOSK FINAL (on exit): {final_result_dict['text']}")
        except json.JSONDecodeError:
            print("\nCould not decode final Vosk result.")
    print("\nScript finished.")

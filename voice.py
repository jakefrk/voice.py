import sounddevice as sd
import numpy as np
import sys
import time
import json
import os
import librosa
import parselmouth
from collections import deque

# --- Configuration Section ---
# Central place for tweaking parameters without digging through the code.

# --- Vosk Configuration ---
# Specifies the path to the Vosk speech recognition model folder.
# Smaller models are faster but less accurate; larger models are slower but more accurate.
# We started with 'small-en-us-0.15' for real-time performance.
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
# --------------------------

# --- Audio Input Configuration ---
DEVICE = None        # `None` uses the default system input device. Can be set to a specific device index if needed.
CHANNELS = 1         # Use Mono audio input, as most speech analysis algorithms expect single-channel data.
DTYPE = 'int16'      # Data type for audio samples. Vosk and Parselmouth often work well with 16-bit integers.
SAMPLE_RATE = None   # Placeholder; will be determined dynamically from the default input device during initialization.
CHUNK_DURATION = 0.5 # Duration of audio chunks processed in each callback (in seconds).
                     # Smaller chunks -> Lower latency for RMS/F0 updates, but potentially less stable analysis per chunk.
                     # Larger chunks -> Higher latency, but more data for analysis algorithms. 0.5s is a compromise.
CHUNK_SIZE = None    # Placeholder; calculated as SAMPLE_RATE * CHUNK_DURATION during initialization.
# --------------------------

# --- Librosa F0 (Pitch) Configuration ---
# Define the expected fundamental frequency range for pitch detection using librosa.yin.
# Helps the algorithm focus and avoid errors (e.g., detecting harmonics as F0).
# 'C2' (~65 Hz) is a common lower bound for human speech.
FMIN = librosa.note_to_hz('C2')
# 'C7' (~2093 Hz) is a high upper bound, chosen because testing showed that
# a narrower range sometimes missed valid pitch changes in this specific setup.
FMAX = librosa.note_to_hz('C7')
# -----------------------------

# --- Analysis Buffering Configuration ---
# These buffers store recent data for analyses performed at the end of an utterance.
# Jitter, Shimmer, Timing, and Inflection analysis benefit from analyzing slightly longer segments.

# Audio buffer for Jitter/Shimmer analysis
# Stores the raw audio chunks.
BUFFER_DURATION_SECONDS = 3 # How many seconds of audio to keep for utterance-level analysis.
                            # 3 seconds provides sufficient data for Parselmouth's Jitter/Shimmer
                            # calculations, which failed on shorter, per-chunk analysis.
MAX_BUFFER_CHUNKS = None    # Calculated during init based on BUFFER_DURATION_SECONDS and CHUNK_DURATION.
audio_buffer = None         # The deque object holding audio chunks.

# F0 buffer for Inflection analysis
# Stores the per-chunk F0 values calculated by librosa.
F0_BUFFER_DURATION_SECONDS = 3 # Duration of F0 history to keep. Matched audio buffer for simplicity.
MAX_F0_BUFFER_CHUNKS = None    # Calculated during init.
f0_buffer = None               # The deque object holding F0 values.
# ----------------------


# --- Vosk Model Loading & Initialization ---
# Load the speech recognition model specified by VOSK_MODEL_PATH.
# This happens once at the start of the script.
try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    # Provide guidance if the Vosk library isn't installed.
    print(json.dumps({"type": "error", "source": "init", "message": "Vosk library not found. Please install via pip: pip install vosk"}), flush=True, file=sys.stderr)
    sys.exit(1)

# Check if the specified model path exists before attempting to load.
if not os.path.exists(VOSK_MODEL_PATH):
    print(json.dumps({"type": "error", "source": "init", "message": f"Vosk model folder not found at '{VOSK_MODEL_PATH}'. Download from https://alphacephei.com/vosk/models"}), flush=True, file=sys.stderr)
    sys.exit(1)

# Load the model from the specified path. Can take a moment depending on model size.
model = Model(VOSK_MODEL_PATH)
# print(f"Loading Vosk model from: {VOSK_MODEL_PATH}") # Keep console logs minimal for JSON output
# print("Vosk model loaded successfully.")
# --------------------------

# --- Initialize Recognizer, Audio Device Info, Buffers ---
# This block queries the audio device, sets the sample rate, initializes the Vosk recognizer
# with that sample rate, and sets up the fixed-size buffers.
try:
    # Query the default input device information.
    default_input_info = sd.query_devices(DEVICE, 'input')
    # Dynamically set the SAMPLE_RATE based on the device's default.
    SAMPLE_RATE = int(default_input_info['default_samplerate'])
    # print(f"Using default input device: {default_input_info['name']}")
    # print(f"Using sample rate: {SAMPLE_RATE} Hz")

    # Initialize the Vosk recognizer instance *after* knowing the sample rate.
    # The recognizer needs the sample rate to interpret the audio data correctly.
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    # Enable word timestamps in the final Vosk result. Crucial for timing/staccato analysis.
    recognizer.SetWords(True)
    # print("Vosk recognizer initialized.")

    # Calculate the actual CHUNK_SIZE in samples based on SAMPLE_RATE and CHUNK_DURATION.
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    # print(f"Using chunk size: {CHUNK_SIZE} samples ({CHUNK_DURATION}s)")

    # Calculate buffer sizes in terms of chunks and initialize the deques.
    # Deques automatically discard oldest elements when full, creating sliding windows.
    MAX_BUFFER_CHUNKS = int(BUFFER_DURATION_SECONDS / CHUNK_DURATION)
    audio_buffer = deque(maxlen=MAX_BUFFER_CHUNKS)
    # print(f"Audio buffer size: {MAX_BUFFER_CHUNKS} chunks ({BUFFER_DURATION_SECONDS}s)")

    MAX_F0_BUFFER_CHUNKS = int(F0_BUFFER_DURATION_SECONDS / CHUNK_DURATION)
    f0_buffer = deque(maxlen=MAX_F0_BUFFER_CHUNKS)
    # print(f"F0 buffer size: {MAX_F0_BUFFER_CHUNKS} chunks ({F0_BUFFER_DURATION_SECONDS}s)")

except Exception as e:
    # Catch any errors during device query or initialization.
    print(json.dumps({"type": "error", "source": "init", "message": f"Error during initialization: {e}"}), flush=True, file=sys.stderr)
    sys.exit(1)
# ------------------------------------

# --- Helper Function Definitions ---
# These functions perform specific analysis tasks on audio data or Vosk results.
# Defining them here keeps the main audio_callback function cleaner.

# Constant for normalizing int16 audio data to float range [-1.0, 1.0]
INT16_MAX = np.iinfo(np.int16).max

def process_audio_chunk_float(audio_chunk_int16):
    """
    Converts an audio chunk from int16 format to float32, normalized to [-1.0, 1.0].
    Librosa functions generally expect float audio data.
    Uses .flatten() to ensure the array is 1D, removing the channel dimension if present.
    """
    return audio_chunk_int16.astype(np.float32).flatten() / INT16_MAX

def calculate_rms(audio_chunk_float32):
    """
    Calculates the Root Mean Square (RMS) energy of a float audio chunk using Librosa.
    RMS is a good proxy for perceived loudness/volume.
    Returns the average RMS value across the chunk.
    """
    # frame_length and hop_length arguments could be added for finer control,
    # but defaults often work well for chunk-level analysis.
    rms = librosa.feature.rms(y=audio_chunk_float32)[0] # librosa.feature.rms returns a 2D array [[rms_values]]
    return np.mean(rms)

def calculate_f0_librosa(audio_chunk_float32, sr):
    """
    Calculates the fundamental frequency (F0/pitch) using Librosa's YIN algorithm.
    YIN is generally robust for speech pitch tracking.
    Filters out frames where F0 is likely unvoiced (f0 <= 0) and calculates the mean
    of the remaining voiced F0 values. Returns 0.0 if no voiced frames are found.
    """
    f0 = librosa.yin(y=audio_chunk_float32, fmin=FMIN, fmax=FMAX, sr=sr)
    # Use nanmean to handle potential NaN values returned by yin for unvoiced frames.
    voiced_f0 = f0[f0 > 0] # Select only positive (voiced) F0 estimates
    return np.nanmean(voiced_f0) if np.any(voiced_f0) else 0.0 # Return mean if any voiced F0 exists, else 0

def calculate_jitter_shimmer(long_audio_chunk_float32, sr):
    """
    Calculates jitter (Relative Average Perturbation - RAP) and shimmer (local, apq5)
    on a longer segment of audio using Parselmouth/Praat.
    Jitter measures cycle-to-cycle frequency variations (voice stability).
    Shimmer measures cycle-to-cycle amplitude variations (voice stability).
    These calculations require a longer audio segment (e.g., 3 seconds) to be reliable,
    as Praat needs sufficient data to establish pitch periods accurately.
    Returns tuple: (jitter_percentage, shimmer_percentage).
    """
    try:
        # Create Parselmouth Sound object. Requires float64 data type.
        snd = parselmouth.Sound(long_audio_chunk_float32.astype(np.float64), sampling_frequency=sr)
        # Calculate pitch using Praat's default autocorrelation method.
        # Using defaults proved more reliable than specifying FMIN/FMAX here during testing.
        pitch = snd.to_pitch_ac()
        voiced_frames = pitch.count_voiced_frames()

        # Praat needs a minimum number of voiced frames to calculate perturbations reliably.
        # 10 is a heuristic threshold; analysis is unlikely to be meaningful below this.
        if voiced_frames < 10:
             # print("  [Debug] Not enough voiced frames for J/S analysis.", file=sys.stderr)
             return 0.0, 0.0

        # Convert the Pitch object to a PointProcess, representing glottal closure instants.
        # This is essential for jitter/shimmer calculations.
        # CRITICAL FIX: Praat command needs BOTH Sound and Pitch objects, passed as a list.
        point_process = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

        # Check if PointProcess creation was successful and contains enough points.
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 2: # Need at least 2 cycles for period/amplitude comparison.
            # print("  [Debug] Not enough points in PointProcess for J/S.", file=sys.stderr)
            return 0.0, 0.0

        # Calculate Jitter (RAP). RAP averages period differences over 3 cycles.
        # Parameters are standard defaults used in Praat for RAP calculation.
        jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100 # Convert to percentage

        # Calculate Shimmer (local, apq5). Measures amplitude perturbation over 5 cycles.
        # CRITICAL FIX: Praat command needs BOTH Sound and PointProcess, passed as a list.
        # Parameters are standard defaults for local shimmer (apq5).
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100 # Convert to percentage

        # Handle potential NaN results if Praat fails internally on specific segments.
        if np.isnan(jitter_rap): jitter_rap = 0.0
        if np.isnan(shimmer_local): shimmer_local = 0.0

        return jitter_rap, shimmer_local # Return the calculated percentages

    except Exception as e:
        # Catch any unexpected errors during the Parselmouth/Praat interaction.
        # Print error to stderr for backend debugging, return safe defaults.
        print(f"Parselmouth calculation error: {e}", file=sys.stderr)
        return 0.0, 0.0

def analyze_timing(result_dict):
    """
    Analyzes word timings from a final Vosk result dictionary.
    Calculates statistics about inter-word pauses and word durations.
    Identifies potential staccato patterns based on these statistics.
    Returns:
        timing_stats (dict): Dictionary containing calculated statistics.
        staccato_flags (list): List of strings describing triggered staccato conditions.
        is_staccato (bool): True if staccato pattern is detected based on flag count.
    """
    words = result_dict.get('result', []) # 'result' key holds the list of word timings

    # Initialize return values
    stats = {
        "avg_pause": 0.0, "max_pause": 0.0, "std_pause": 0.0,
        "avg_word_dur": 0.0, "std_word_dur": 0.0, "speech_rate_wps": 0.0
    }
    staccato_flags = []
    is_staccato = False

    if not words: return stats, staccato_flags, is_staccato # Nothing to analyze

    pauses = []
    durs = []
    # Use the start time of the first word as the initial reference point.
    # Ensure conversion to float for calculations.
    first_start = float(words[0].get('start', 0.0))
    last_end = first_start # Initialize last_end relative to the start of the utterance

    for i, w in enumerate(words):
        start_time = float(w.get('start', -1))
        end_time = float(w.get('end', -1))
        # Skip word if timing info is invalid
        if start_time < 0 or end_time < 0 or end_time < start_time: continue

        # Calculate word duration
        d = end_time - start_time
        durs.append(d)

        # Calculate pause before the current word (if not the first word)
        if i > 0:
             p = start_time - last_end # Time difference between previous word end and current word start
             if p < 0: p = 0.0 # Ensure pause is not negative due to minor timing overlaps
             pauses.append(p)

        last_end = end_time # Update end time for the next iteration

    # Compute statistics safely, handling cases where lists might be empty
    avg_p = np.mean(pauses) if pauses else 0.0
    max_p = np.max(pauses) if pauses else 0.0
    std_p = np.std(pauses) if pauses else 0.0
    avg_d = np.mean(durs) if durs else 0.0
    std_d = np.std(durs) if durs else 0.0
    # Total duration from the start of the first word to the end of the last word
    total_t = last_end - first_start
    # Calculate speech rate (words per second), avoid division by zero for very short utterances
    rate = len(words) / total_t if total_t > 0.1 else 0.0

    # --- Calculate Statistics ---
    # ... (calculations for avg_p, max_p, std_p, avg_d, std_d, rate) ...

    # --- Store rounded statistics, CASTING TO FLOAT ---
    stats = {
        "avg_pause": float(round(avg_p, 3)),
        "max_pause": float(round(max_p, 3)),
        "std_pause": float(round(std_p, 3)),
        "avg_word_dur": float(round(avg_d, 3)),
        "std_word_dur": float(round(std_d, 3)),
        "speech_rate_wps": float(round(rate, 2))
    }

    # --- Staccato Detection Logic ---
    # These rules aim to identify patterns characteristic of staccato speech.
    # Thresholds are based on observation during testing and likely need tuning
    # for different users or use cases.

    # 1) Long Pause: Is the longest pause both absolutely long (e.g., >1s) and
    #    relatively long compared to the average pause (e.g., >3x)?
    #    Helps flag utterances with significant, isolated hesitations.
    LONG_PAUSE_ABS_THRESHOLD = 1.0 # seconds
    LONG_PAUSE_REL_FACTOR = 3.0
    if pauses and max_p > LONG_PAUSE_ABS_THRESHOLD and max_p > avg_p * LONG_PAUSE_REL_FACTOR and avg_p > 0:
        staccato_flags.append(f"LongPause({max_p:.2f}s)")

    # 2) Variable Pauses: Is the standard deviation of pauses high relative to the
    #    average pause duration (e.g., > 80% of the average)?
    #    Indicates inconsistent timing between words, characteristic of uneven rhythm.
    VAR_PAUSE_REL_THRESHOLD = 0.8
    if pauses and avg_p > 0 and std_p > avg_p * VAR_PAUSE_REL_THRESHOLD:
        staccato_flags.append(f"VarPauses({std_p:.2f}s)")

    # 3) Fast Rate: Is the overall speech rate unusually high (e.g., >4 wps)?
    #    Could indicate rushed speech or short bursts. Normal conversational English is often ~2-3 wps.
    FAST_RATE_THRESHOLD = 4.0 # words per second
    if rate > FAST_RATE_THRESHOLD:
        staccato_flags.append(f"FastRate({rate:.1f}wps)")

    # Note: Removed flags for SlowRate and VariableWordDur from previous iteration
    # as they were less directly indicative of the intended "staccato" definition,
    # but they could be added back if desired.

    # --- Final Staccato Decision ---
    # Require multiple flags (e.g., >= 2) to trigger a staccato alert.
    # This reduces sensitivity to minor variations and requires stronger evidence.
    STACCATO_FLAG_THRESHOLD = 2
    is_staccato = len(staccato_flags) >= STACCATO_FLAG_THRESHOLD

    # Print simple summary to console (optional, can be removed for pure JSON output)
    # print(f"  [Timing] Stats: AvgP={stats['avg_pause']} MaxP={stats['max_pause']} StdP={stats['std_pause']} Rate={stats['speech_rate_wps']}")
    # if is_staccato: print(f"  [Staccato Alert] Flags: {', '.join(flags)}")

    return stats, staccato_flags, is_staccato # Return all computed information


def analyze_inflection(f0_sequence, sample_rate, chunk_duration):
    """
    Analyzes the F0 sequence from the end of an utterance for upward inflection.
    Calculates the linear trend (slope) of the F0 contour over the last part
    of the utterance. A positive slope indicates rising pitch.
    Returns:
        slope (float): The calculated F0 slope (Hz per chunk index).
        is_upward (bool): True if the slope exceeds the upward inflection threshold.
    """
    slope = 0.0
    is_upward = False

    if not f0_sequence: return slope, is_upward # Cannot analyze empty sequence

    f0_array = np.array(f0_sequence) # Convert deque to numpy array

    # Define the duration at the end of the utterance to analyze for inflection.
    # 1.25s (approx 2 chunks of 0.5s + overlap) was found necessary during testing
    # to reliably get enough data points for slope calculation.
    analysis_duration_s = 1.25
    num_chunks_to_analyze = int(analysis_duration_s / chunk_duration)
    # Ensure we don't try to analyze more chunks than available in the buffer.
    num_chunks_to_analyze = min(num_chunks_to_analyze, len(f0_array))

    # Require at least 2 data points to fit a line for the slope.
    if num_chunks_to_analyze < 2:
        # print("  [Inflection] Not enough data points for trend analysis.")
        return slope, is_upward

    # Extract the F0 values for the final segment of the utterance.
    final_f0_segment = f0_array[-num_chunks_to_analyze:]
    # Filter out unvoiced frames (F0 <= 0) as they don't represent pitch.
    voiced_f0 = final_f0_segment[final_f0_segment > 0]

    # Need at least 2 *voiced* data points in the final segment to calculate slope.
    if len(voiced_f0) < 2:
        # print("  [Inflection] Not enough voiced data points in final segment.")
        return slope, is_upward

    # --- Calculate Slope using Linear Regression ---
    indices = np.arange(len(voiced_f0))
    try:
        slope_np, intercept = np.polyfit(indices, voiced_f0, 1)
        # Ensure slope is standard Python float before returning
        slope = float(round(slope_np, 2))

        # --- Upward Inflection Decision ---
        # Define a threshold for what constitutes a significant upward slope.
        # This value is highly sensitive and needs tuning based on perceived inflection.
        # A value of 1.5 means the pitch needs to rise, on average, by 1.5 Hz
        # for each step forward in the chunk indices within the voiced segment.
        UPWARD_SLOPE_THRESHOLD = 1.5
        if slope > UPWARD_SLOPE_THRESHOLD:
            is_upward = True

        # Optional console print for debugging slope value
        # print(f"  [Inflection] Analysis Segment F0 Slope: {slope} Hz/chunk_idx")
        # if is_upward: print(f"  [Inflection Alert] Potential upward inflection detected (Slope={slope})")

    except np.linalg.LinAlgError:
        # Catch errors during the polyfit calculation (e.g., if data is degenerate).
        print("  [Inflection] Could not calculate slope (linear algebra error).", file=sys.stderr)
        slope = 0.0 # Ensure slope is float even on error

    return slope, is_upward # Return standard float and boolean

# ------------------------------------

# === Main Audio Processing Callback ===
# This function is executed by the sounddevice library in a separate thread
# for every new chunk of audio data received from the microphone.
def audio_callback(indata, frames, time_info, status):
    """
    Processes each incoming audio chunk: extracts features, buffers data,
    runs utterance-level analysis via Vosk, and outputs results as JSON.
    """
    # Make global variables accessible within the callback
    global recognizer, SAMPLE_RATE, audio_buffer, f0_buffer, CHUNK_DURATION

    # Check for audio stream errors reported by sounddevice.
    if status:
        error_payload = {"type": "error", "source": "audio_callback", "message": f"Audio stream status: {status}"}
        print(json.dumps(error_payload), flush=True, file=sys.stderr)
        return

    # 1. Buffer Raw Audio: Store the current int16 chunk for later utterance analysis.
    #    Use .copy() to ensure the buffer stores independent data, not just a reference.
    audio_buffer.append(indata.copy())

    # 2. Per-Chunk Feature Extraction: Calculate features immediately available from the current chunk.
    #    Convert to float first as required by librosa.
    audio_float32 = process_audio_chunk_float(indata)
    current_rms = calculate_rms(audio_float32)       # Calculate volume proxy
    current_f0 = calculate_f0_librosa(audio_float32, SAMPLE_RATE) # Calculate pitch

    # 3. Buffer F0: Store the calculated F0 value for later inflection analysis.
    f0_buffer.append(current_f0)

    # --- Output Per-Chunk JSON ---
    # Send immediate feedback features (RMS, F0) to the frontend via stdout JSON.
    # Rounding values helps reduce JSON message size.
    chunk_payload = {
        "type": "chunk_features",
        # Cast NumPy floats to standard Python floats
        "rms": float(round(current_rms, 4)),
        "f0": float(round(current_f0, 2))
    }
    # Use flush=True to ensure the output is sent immediately without buffering by Python.
    print(json.dumps(chunk_payload), flush=True)

    # 4. Vosk Speech Recognition: Feed the audio chunk (as bytes) to the recognizer.
    #    `AcceptWaveform` returns True if Vosk detects the end of an utterance.
    if recognizer.AcceptWaveform(indata.tobytes()):
        # --- Utterance End Detected ---
        result_json = recognizer.Result() # Get the final recognition result (JSON string)
        try:
            # Parse the JSON string into a Python dictionary.
            result_dict = json.loads(result_json)
        except json.JSONDecodeError:
            # Handle cases where Vosk might return invalid JSON.
            print(json.dumps({"type": "error", "source": "vosk", "message": "Failed to decode JSON result"}), flush=True, file=sys.stderr)
            result_dict = {} # Use empty dict to prevent errors below

        final_text = result_dict.get('text', '') # Extract the transcribed text

        # Proceed only if transcription is not empty.
        if final_text:
            # --- Perform Utterance-Level Analyses ---
            # These analyses use the buffered data collected over the last few seconds.

            # Calculate Jitter & Shimmer using the buffered audio
            segment_jitter = 0.0
            segment_shimmer = 0.0
            if len(audio_buffer) > 1: # Ensure buffer has enough data
                buffered_int16_data = np.concatenate(list(audio_buffer), axis=0)
                buffered_float32_data = process_audio_chunk_float(buffered_int16_data)
                segment_jitter, segment_shimmer = calculate_jitter_shimmer(buffered_float32_data, SAMPLE_RATE)

            # Analyze Timing/Staccato using the Vosk word result timings
            timing_stats = {}
            staccato_flags = []
            is_staccato = False
            if 'result' in result_dict: # Check if word timings ('result' key) exist
                timing_stats, staccato_flags, is_staccato = analyze_timing(result_dict)

            # Analyze Inflection using the buffered F0 values
            inflection_slope = 0.0
            is_upward_inflection = False
            if f0_buffer: # Ensure F0 buffer is not empty
                 inflection_slope, is_upward_inflection = analyze_inflection(list(f0_buffer), SAMPLE_RATE, CHUNK_DURATION)

            # --- Construct and Output Utterance JSON Payload ---
            # Bundle all utterance-level analysis results into a single JSON object.
            utterance_payload = {
                "type": "utterance_analysis",
                "text": final_text,
                # Cast NumPy floats to standard Python floats
                "jitter": float(round(segment_jitter, 2)),
                "shimmer": float(round(segment_shimmer, 2)),
                "timing_stats": timing_stats, # Ensure timing_stats values are also floats below
                "staccato_flags": staccato_flags,
                "is_staccato": is_staccato,
                # Cast NumPy float to standard Python float
                "inflection_slope": float(inflection_slope),
                "is_upward_inflection": is_upward_inflection
            }
            # Print the combined utterance analysis results to stdout.
            print(json.dumps(utterance_payload), flush=True)

            # Optional: Clear the F0 buffer after each utterance analysis?
            # If cleared, inflection analysis only considers the current utterance.
            # If not cleared, it uses a sliding window of the last N seconds of F0 history.
            # Let's keep the sliding window approach for now.
            # f0_buffer.clear()

    # Handle partial results (text recognized mid-utterance)
    else:
        partial_json = recognizer.PartialResult() # Get partial result JSON string
        partial_dict = json.loads(partial_json)
        partial_text = partial_dict.get('partial', '')
        # If partial text exists, send it as a separate JSON message type.
        if partial_text:
             partial_payload = {"type": "partial_text", "text": partial_text}
             print(json.dumps(partial_payload), flush=True)

# === Main Execution Block ===
# This code runs when the script is executed directly.
try:
    # Send a status message indicating the script has started.
    print(json.dumps({"type": "status", "message": "Voice analysis script started."}), flush=True)

    # --- Start Audio Stream ---
    # Create an input audio stream using sounddevice.
    # This runs in the background, continuously capturing audio and calling
    # the `audio_callback` function for each chunk.
    with sd.InputStream(
        device=DEVICE,          # Use default input device
        channels=CHANNELS,      # Mono audio
        samplerate=SAMPLE_RATE, # Determined from device
        blocksize=CHUNK_SIZE,   # Size of chunks passed to callback
        dtype=DTYPE,          # Data type (int16)
        callback=audio_callback # Function to call for each chunk
    ):
        # Keep the main thread alive while the background audio stream runs.
        # The actual processing happens in the callback thread.
        while True:
            time.sleep(0.1) # Sleep briefly to prevent busy-waiting and high CPU usage

# Handle graceful shutdown on Ctrl+C.
except KeyboardInterrupt:
    print(json.dumps({"type": "status", "message": "Stopping analysis (KeyboardInterrupt)."}), flush=True)
# Handle any other unexpected errors during streaming.
except Exception as e:
    print(json.dumps({"type": "error", "source": "main_loop", "message": f"An error occurred: {e}"}), flush=True, file=sys.stderr)
    sys.exit(1)
# Final cleanup actions.
finally:
    # Attempt to get any final partial result from Vosk upon exit.
    # This might capture words spoken just before Ctrl+C was pressed.
    if 'recognizer' in locals() and hasattr(recognizer, 'FinalResult'):
        final_result_json = recognizer.FinalResult()
        try:
            final_result_dict = json.loads(final_result_json)
            final_text = final_result_dict.get('text', '')
            if final_text:
                 # Send this final fragment as a standard utterance payload for consistency
                 # Note: Analysis on this final fragment might be incomplete as buffers might not represent it fully.
                 # Consider if special handling is needed or if just sending the text is sufficient.
                 final_payload = {"type": "utterance_analysis", "text": final_text, "is_final_fragment": True} # Add flag
                 print(json.dumps(final_payload), flush=True)
        except json.JSONDecodeError:
            # Ignore errors decoding the very final result fragment.
            pass
    # Send a final status message indicating the script has finished.
    print(json.dumps({"type": "status", "message": "Voice analysis script finished."}), flush=True)

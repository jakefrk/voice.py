// renderer.js - Electron Renderer Process Logic

console.log('Renderer script loaded.');

// --- DOM Element References ---
// const topStatusDiv = document.getElementById('top-status'); // Removed
const stopButton = document.getElementById('stopButton');
const rmsValueSpan = document.getElementById('rms-value');
const f0ValueSpan = document.getElementById('f0-value');
const wordValueSpan = document.getElementById('word-value');
const textValueSpan = document.getElementById('text-value');
const shakinessValueSpan = document.getElementById('shakiness-value');
const timingValueSpan = document.getElementById('timing-value');
const inflectionValueSpan = document.getElementById('inflection-value');
const statusIndicator = document.getElementById('status-indicator');
const statusIcon = document.getElementById('status-icon'); // This span will now hold text
// const statusText = document.getElementById('status-text'); // Removed

// Feedback Bar References
const inflectionBar = document.getElementById('inflection-bar');
const staccatoBar = document.getElementById('staccato-bar');
const shakinessBar = document.getElementById('shakiness-bar');
// const trailingOffBar = document.getElementById('trailing-off-bar'); // Add later

// --- State & Config ---
let isListening = false;
// let alertTimeout = null; // Removed
let pulseTimeout = null;
// let lastPartialText = ''; // No longer needed for pulse
const RMS_PULSE_MIN_THRESHOLD = 0.005; // ** TUNE ** Minimum RMS to consider for pulsing (avoids noise)
const RMS_INCREASE_FACTOR = 1.5;   // ** TUNE ** How much RMS needs to increase relative to previous chunk to pulse (e.g., 1.5 = 50% increase)
let previousRMS = 0.0; // Track previous RMS value
const ALERT_DURATION_MS = 2000; // How long alert stays red

// Store timeouts for each bar alert reversion
let barAlertTimeouts = {
    inflection: null,
    staccato: null,
    shakiness: null
    // trailing_off: null
};

// --- UI Update Functions ---
// updateTopStatus function removed

// Modified: Handles circle background color AND text inside circle
function setStatusIndicator(statusClass, statusText) {
    if (!statusIndicator || !statusIcon) return;
    // Set background color class
    statusIndicator.classList.remove('status-ok', 'status-listening', 'status-error', 'status-stopped');
    statusIndicator.classList.add(statusClass);
    // Set text content
    statusIcon.textContent = statusText;
}

// Function to set feedback bars to a specific state (neutral, ok, alert)
function setFeedbackBarsState(stateClass) {
    const bars = [inflectionBar, staccatoBar, shakinessBar /*, trailingOffBar */];
    bars.forEach(bar => {
        if (bar) {
            bar.classList.remove('neutral', 'ok', 'alert');
            bar.classList.add(stateClass);
        }
    });
    // Clear any pending alert timeouts when setting a general state
    Object.keys(barAlertTimeouts).forEach(key => {
         if (barAlertTimeouts[key]) clearTimeout(barAlertTimeouts[key]);
         barAlertTimeouts[key] = null;
    });
}

// Updated state update function
function setListeningVisuals(listening) {
    isListening = listening;
    stopButton.disabled = !listening;
    setFeedbackBarsState(listening ? 'ok' : 'neutral'); // Set bars state

    if (listening) {
        // updateTopStatus('Listening...'); // Removed
        setStatusIndicator('status-listening', 'Listening'); // Update circle color and text
        previousRMS = 0.0;
        // Reset text values on start
        textValueSpan.textContent = '"--"';
        wordValueSpan.textContent = '"--"';
    } else {
        // updateTopStatus('Stopped.'); // Removed
        setStatusIndicator('status-stopped', 'Stopped'); // Update circle color and text
        stopButton.disabled = true;
        previousRMS = 0.0;
    }
}

// displayVisualAlert function removed

// Function to trigger the pulse effect
function triggerPulse() {
    if (!statusIndicator || !isListening) return;
    if (pulseTimeout) clearTimeout(pulseTimeout);
    statusIndicator.classList.add('pulse');
    pulseTimeout = setTimeout(() => {
        statusIndicator.classList.remove('pulse');
        pulseTimeout = null;
    }, 150);
}

// Function to update a specific feedback bar's state
function updateFeedbackBar(barElement, isAlert) {
    if (!barElement) return;
    barElement.classList.remove('alert', 'ok'); // Clear previous states
    if (isAlert) {
        barElement.classList.add('alert');
    } else {
         barElement.classList.add('ok'); // Indicate OK status
    }
}

// --- Bar Alert Logic ---
function triggerBarAlert(barElement, barKey) {
    if (!barElement || !isListening) return; // Only alert if listening

    // Clear existing timeout for this specific bar
    if (barAlertTimeouts[barKey]) {
        clearTimeout(barAlertTimeouts[barKey]);
    }

    // Set to alert state
    barElement.classList.remove('ok', 'neutral');
    barElement.classList.add('alert');

    // Set timeout to revert to 'ok' state
    barAlertTimeouts[barKey] = setTimeout(() => {
        if (isListening) { // Check if still listening before reverting
             barElement.classList.remove('alert');
             barElement.classList.add('ok');
        }
        barAlertTimeouts[barKey] = null; // Clear timeout ID
    }, ALERT_DURATION_MS);
}

// --- Event Listeners ---
stopButton.addEventListener('click', () => {
    console.log('Quit button clicked');
    stopButton.disabled = true;
    // updateTopStatus('Quitting...'); // Removed
    setStatusIndicator('status-stopped', 'Quitting...'); // Update circle text
    if (pulseTimeout) clearTimeout(pulseTimeout);
    statusIndicator.classList.remove('pulse');
    setFeedbackBarsState('neutral');
    window.electronAPI.send('quit-app');
});

// --- IPC Handlers ---
window.electronAPI.on('python-data', (data) => {
    if (!isListening) {
        setListeningVisuals(true);
    }

    // Ensure status indicator is 'Listening' if we are receiving data
    if (statusIndicator && !statusIndicator.classList.contains('status-listening')) {
        setStatusIndicator('status-listening', 'Listening');
    }

    switch (data.type) {
        case 'chunk_features':
            const currentRMS = data.rms;
            rmsValueSpan.textContent = currentRMS.toFixed(4);
            f0ValueSpan.textContent = `${data.f0.toFixed(2)} Hz`;

            // --- Pulse on RMS Increase (Syllable Proxy) ---
            if (currentRMS > RMS_PULSE_MIN_THRESHOLD && currentRMS > (previousRMS * RMS_INCREASE_FACTOR)) {
                triggerPulse();
            }
            previousRMS = currentRMS;
            break;
        case 'utterance_analysis':
            // Update sentence with quotes
            textValueSpan.textContent = data.text ? `"${data.text}"` : '"--"';
            // Clear word field with quotes
            wordValueSpan.textContent = '"--"';
            // Update stats fields
            shakinessValueSpan.textContent = `J: ${data.jitter.toFixed(2)}% | S: ${data.shimmer.toFixed(2)}%`;
            timingValueSpan.textContent = `Rate: ${data.timing_stats?.speech_rate_wps} wps | Max Pause: ${data.timing_stats?.max_pause}s`;
            inflectionValueSpan.textContent = `Slope: ${data.inflection_slope.toFixed(2)}`;

            // --- Trigger Bar Alerts ---
            if (data.is_upward_inflection) {
                triggerBarAlert(inflectionBar, 'inflection');
            }
            if (data.is_staccato) {
                triggerBarAlert(staccatoBar, 'staccato');
            }

            const JITTER_THRESHOLD = 2.5; // Example threshold - TUNE!
            const SHIMMER_THRESHOLD = 18.0; // Example threshold - TUNE!
            const isShaky = data.jitter > JITTER_THRESHOLD || data.shimmer > SHIMMER_THRESHOLD;
            if (isShaky) {
                triggerBarAlert(shakinessBar, 'shakiness');
            }

            // TODO: Add trailing off logic later

            previousRMS = 0.0; // Reset previous RMS after utterance
            break;
        case 'partial_text':
             // Update individual word display only
             const newText = data.text;
             if (newText) { // Check if text exists
                 const words = newText.trim().split(' '); // Trim whitespace before splitting
                 const lastWord = words[words.length - 1];
                 if (lastWord) {
                      // Update word with quotes
                      wordValueSpan.textContent = `"${lastWord}"`;
                      // Pulse logic moved to chunk_features
                 }
             }
             break;
    }
});

window.electronAPI.on('python-error', (errorMsg) => {
    console.error('Received error from Python backend:', errorMsg);
    if (pulseTimeout) clearTimeout(pulseTimeout);
    statusIndicator.classList.remove('pulse');
    // updateTopStatus(`Error: ${errorMsg.split('\n')[0]}`); // Removed
    setStatusIndicator('status-error', 'Error!'); // Set error state in circle
    setListeningVisuals(false); // Set stopped state (resets bars to neutral)
});

window.electronAPI.on('python-status', (statusInfo) => {
     console.log('Received status from Python backend:', statusInfo);
     if (statusInfo.status === 'stopped') {
         if (pulseTimeout) clearTimeout(pulseTimeout);
         statusIndicator.classList.remove('pulse');
         // updateTopStatus('Backend stopped.'); // Removed
         setStatusIndicator('status-stopped', 'Stopped'); // Update circle state/text
         setListeningVisuals(false); // Set stopped state (resets bars to neutral)
     }
});

// --- Initialization ---
function initializeApp() {
    // updateTopStatus('Starting analysis...'); // Removed
    setStatusIndicator('status-listening', 'Starting...'); // Set initial circle state/text
    stopButton.disabled = false;
    isListening = true;
    setListeningVisuals(true); // Set initial visual state correctly
    window.electronAPI.send('start-analysis');
}

initializeApp();

// renderer.js - Electron Renderer Process Logic

console.log('Renderer script loaded.');

// --- DOM Element References ---
// const topStatusDiv = document.getElementById('top-status'); // Removed
const stopButton = document.getElementById('stopButton');
const rmsValueSpan = document.getElementById('rms-value');
const f0ValueSpan = document.getElementById('f0-value');
const wordValueSpan = document.getElementById('word-value');
const textValueSpan = document.getElementById('text-value');
const shakinessValueSpan = document.getElementById('shakiness-value'); // Re-added
// const jitterValueSpan = document.getElementById('jitter-value'); // Removed
// const shimmerValueSpan = document.getElementById('shimmer-value'); // Removed
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

// *** ADDED Detail Element References ***
const inflectionDetail = document.getElementById('inflection-detail');
const staccatoDetail = document.getElementById('staccato-detail');
const shakinessDetail = document.getElementById('shakiness-detail');

// --- State & Config ---
let isListening = false;
// let alertTimeout = null; // Removed
// let pulseTimeout = null; // Removed
// let lastPartialText = ''; // No longer needed for pulse
// const RMS_PULSE_MIN_THRESHOLD = 0.005; // Removed
// const RMS_INCREASE_FACTOR = 1.5; // Removed
// let previousRMS = 0.0; // Removed
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
    // Stop CSS animation if not listening
    statusIndicator.style.animation = (statusClass === 'status-listening') ? '' : 'none';
}

// Updated: Now also clears detail text
function setFeedbackBarsState(stateClass) {
    const bars = [inflectionBar, staccatoBar, shakinessBar /*, trailingOffBar */];
    const details = [inflectionDetail, staccatoDetail, shakinessDetail]; // Added details array
    bars.forEach(bar => {
        if (bar) {
            bar.classList.remove('neutral', 'ok', 'alert');
            bar.classList.add(stateClass);
        }
    });
    // *** Clear Detail Text ***
    details.forEach(detail => {
        if (detail) detail.innerHTML = '&nbsp;'; // Use &nbsp; to maintain height
    });
    // Clear any pending alert timeouts
    Object.keys(barAlertTimeouts).forEach(key => {
         if (barAlertTimeouts[key]) clearTimeout(barAlertTimeouts[key]);
         barAlertTimeouts[key] = null;
    });
}

// Updated state update function
function setListeningVisuals(listening) {
    isListening = listening;
    stopButton.disabled = !listening;
    setFeedbackBarsState(listening ? 'ok' : 'neutral'); // Set bars state (also clears details)

    if (listening) {
        // updateTopStatus('Listening...'); // Removed
        setStatusIndicator('status-listening', 'Listening'); // Update circle color and text
        // previousRMS = 0.0; // Removed
        // Reset text values on start
        textValueSpan.textContent = '"--"';
        wordValueSpan.textContent = '"--"';
        // Reset individual Jitter/Shimmer values on start
        shakinessValueSpan.textContent = 'J: --% | S: --%'; // Reset combined value
    } else {
        // updateTopStatus('Stopped.'); // Removed
        setStatusIndicator('status-stopped', 'Stopped'); // Update circle color and text
        stopButton.disabled = true;
        // previousRMS = 0.0; // Removed
    }
}

// displayVisualAlert function removed

// Function to trigger the pulse effect
// triggerPulse function removed

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

// Modified: No longer updates barElement, just manages timeout
function triggerBarAlert(barElement, barKey) {
    if (!barElement || !isListening) return;

    // Clear existing timeout for this specific bar
    if (barAlertTimeouts[barKey]) {
        clearTimeout(barAlertTimeouts[barKey]);
    }
    // Set to alert state (done in utterance_analysis now)
    // barElement.classList.remove('ok', 'neutral');
    // barElement.classList.add('alert');

    // Set timeout to revert to 'ok' state
    barAlertTimeouts[barKey] = setTimeout(() => {
        if (isListening && barElement.classList.contains('alert')) { // Check if still listening and alert is active
             barElement.classList.remove('alert');
             barElement.classList.add('ok');
             // Optionally clear detail text when bar reverts? Or leave it until next utterance? Let's leave it.
        }
        barAlertTimeouts[barKey] = null;
    }, ALERT_DURATION_MS);
}

// --- Event Listeners ---
stopButton.addEventListener('click', () => {
    console.log('Quit button clicked');
    stopButton.disabled = true;
    // updateTopStatus('Quitting...'); // Removed
    setStatusIndicator('status-stopped', 'Quitting...'); // Update circle text
    // if (pulseTimeout) clearTimeout(pulseTimeout); // Removed
    // statusIndicator.classList.remove('pulse'); // Removed
    setFeedbackBarsState('neutral'); // Reset bars and clear details
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
            // if (currentRMS > RMS_PULSE_MIN_THRESHOLD && currentRMS > (previousRMS * RMS_INCREASE_FACTOR)) {
            //     triggerPulse();
            // }
            // previousRMS = currentRMS;
            break;
        case 'utterance_analysis':
            // --- Reset Details First ---
            inflectionDetail.innerHTML = '&nbsp;';
            staccatoDetail.innerHTML = '&nbsp;';
            shakinessDetail.innerHTML = '&nbsp;';

            // Update sentence with quotes
            textValueSpan.textContent = data.text ? `"${data.text}"` : '"--"';
            // Clear word field with quotes
            wordValueSpan.textContent = '"--"';
            // Update stats fields
            shakinessValueSpan.textContent = `J: ${data.jitter.toFixed(2)}% | S: ${data.shimmer.toFixed(2)}%`;
            timingValueSpan.textContent = `Rate: ${data.timing_stats?.speech_rate_wps} wps | Max Pause: ${data.timing_stats?.max_pause}s`;
            inflectionValueSpan.textContent = `Slope: ${data.inflection_slope.toFixed(2)}`;

            // --- Update Feedback Bars and Details ---
            // Inflection
            if (data.is_upward_inflection && inflectionDetail && data.inflection_detail) {
                inflectionBar.classList.remove('ok', 'neutral');
                inflectionBar.classList.add('alert');
                inflectionDetail.textContent = `End words: ${data.inflection_detail.join(', ')}`;
                triggerBarAlert(inflectionBar, 'inflection'); // Start revert timer
            } else if (inflectionBar) {
                 inflectionBar.classList.remove('alert', 'neutral');
                 inflectionBar.classList.add('ok'); // Explicitly set OK
            }

            // Staccato
            if (data.is_staccato && staccatoBar && staccatoDetail) {
                staccatoBar.classList.remove('ok', 'neutral');
                staccatoBar.classList.add('alert');
                let staccatoMsg = '';
                // Prioritize showing words after long pauses if available
                if (data.staccato_detail?.long_pause_words?.length > 0) {
                     staccatoMsg = `Word(s) after long pause: "${data.staccato_detail.long_pause_words.join('", "')}"`;
                } else {
                    // Fallback to general flag descriptions if no long pause words found
                    // Filter flags to provide slightly better descriptions
                    let generalFlags = [];
                    if (data.staccato_flags.some(flag => flag.startsWith('VarPauses'))) {
                        generalFlags.push("High pause variability");
                    }
                    if (data.staccato_flags.some(flag => flag.startsWith('FastRate'))) {
                        generalFlags.push("High speech rate");
                    }
                    staccatoMsg = generalFlags.length > 0 ? generalFlags.join(' & ') : "Staccato pattern detected"; // Default if flags are unexpected
                }
                staccatoDetail.textContent = staccatoMsg;
                triggerBarAlert(staccatoBar, 'staccato'); // Start revert timer
            } else if (staccatoBar) {
                staccatoBar.classList.remove('alert', 'neutral');
                staccatoBar.classList.add('ok'); // Explicitly set OK
            }

            // Shakiness
            const JITTER_THRESHOLD = 2.5;
            const SHIMMER_THRESHOLD = 18.0;
            const isShaky = data.jitter > JITTER_THRESHOLD || data.shimmer > SHIMMER_THRESHOLD;
            if (isShaky && shakinessDetail) {
                 shakinessBar.classList.remove('ok', 'neutral');
                 shakinessBar.classList.add('alert');
                 shakinessDetail.textContent = `J: ${data.jitter}%, S: ${data.shimmer}%`; // Show values
                 triggerBarAlert(shakinessBar, 'shakiness'); // Start revert timer
            } else if (shakinessBar) {
                 shakinessBar.classList.remove('alert', 'neutral');
                 shakinessBar.classList.add('ok'); // Explicitly set OK
            }

            // previousRMS = 0.0; // Removed
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
    // if (pulseTimeout) clearTimeout(pulseTimeout); // Removed
    // statusIndicator.classList.remove('pulse'); // Removed
    setStatusIndicator('status-error', 'Error!'); // Set error state in circle
    setListeningVisuals(false); // Set stopped state (resets bars to neutral)
});

window.electronAPI.on('python-status', (statusInfo) => {
     console.log('Received status from Python backend:', statusInfo);
     if (statusInfo.status === 'stopped') {
         // if (pulseTimeout) clearTimeout(pulseTimeout); // Removed
         // statusIndicator.classList.remove('pulse'); // Removed
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

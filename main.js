// main.js - Electron Main Process

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('node:path');
const { spawn } = require('node:child_process'); // To run Python script

let mainWindow;
let pythonProcess = null; // Keep track of the Python process

function createWindow() {
    // Create the browser window.
    mainWindow = new BrowserWindow({
        width: 800,
        height: 700,
        webPreferences: {
            // Important: Enable Node integration in the renderer process
            // nodeIntegration: true, // Deprecated and potentially insecure
            // contextIsolation: false, // Deprecated and potentially insecure
            // Use a preload script for secure IPC communication
            preload: path.join(__dirname, 'preload.js') // We'll create this next
        }
    });

    // Load the index.html of the app.
    mainWindow.loadFile('index.html');

    // Open the DevTools (optional, useful for debugging)
    // mainWindow.webContents.openDevTools();

    // Handle window closed event
    mainWindow.on('closed', () => {
        mainWindow = null;
        // Python process killed by 'will-quit' or 'window-all-closed'
    });
}

// --- Python Process Management ---

function startPythonProcess() {
    if (pythonProcess) {
        console.log('Python process already running.');
        return;
    }
    console.log('Starting Python process...');

    // Path to your Python executable within the virtual environment
    // Adjust this path based on your OS and venv location!
    // On macOS/Linux: path/to/your/project/venv/bin/python
    // On Windows: path\to\your\project\venv\Scripts\python.exe
    const pythonExecutable = path.join(__dirname, 'venv', 'bin', 'python'); // Adjust if venv name/location differs
    const scriptPath = path.join(__dirname, 'voice.py');

    // Spawn the Python script
    // Use '-u' for unbuffered output, critical for real-time reading
    pythonProcess = spawn(pythonExecutable, ['-u', scriptPath]);

    // Handle stdout data (JSON messages from Python)
    pythonProcess.stdout.on('data', (data) => {
        const dataStr = data.toString().trim();
        // Handle potentially multiple JSON objects per data chunk
        dataStr.split('\n').forEach(line => {
             if (line) {
                try {
                    const jsonData = JSON.parse(line);
                    // console.log('Received data from Python:', jsonData); // Debug print
                    // Send data to the renderer process
                    if (mainWindow && mainWindow.webContents) {
                        mainWindow.webContents.send('python-data', jsonData);
                    }
                } catch (e) {
                    console.error('Error parsing JSON from Python:', e);
                    console.error('Received line:', line);
                }
             }
        });
    });

    // Handle stderr data (errors from Python)
    pythonProcess.stderr.on('data', (data) => {
        const errorMsg = data.toString();
        console.error(`Python stderr: ${errorMsg}`);
        // Optionally send error info to renderer process too
        if (mainWindow && mainWindow.webContents) {
            mainWindow.webContents.send('python-error', errorMsg);
        }
    });

    // Handle process exit
    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null; // Reset the variable
        // Optionally notify renderer process
        if (mainWindow && mainWindow.webContents) {
            mainWindow.webContents.send('python-status', { status: 'stopped', code });
        }
    });

     // Handle spawn errors
    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        pythonProcess = null;
        if (mainWindow && mainWindow.webContents) {
             mainWindow.webContents.send('python-error', `Failed to start Python process: ${err.message}`);
        }
    });
}

function killPythonProcess() {
    if (pythonProcess) {
        console.log('Killing Python process...');
        pythonProcess.kill('SIGTERM'); // Send termination signal
        // Consider SIGKILL if SIGTERM doesn't work after a timeout
        pythonProcess = null;
    }
}

// --- IPC Handlers ---
// Listen for messages from the renderer process

ipcMain.on('start-analysis', () => {
    console.log('IPC: Received start-analysis signal.');
    startPythonProcess();
});

ipcMain.on('stop-analysis', () => {
    console.log('IPC: Received stop-analysis signal (killing python).');
    killPythonProcess();
});

// *** ADD Handler for Quit Button ***
ipcMain.on('quit-app', () => {
    console.log('IPC: Received quit-app signal.');
    // killPythonProcess(); // Python process is killed by 'will-quit' anyway
    app.quit(); // Quit the entire application
});

// --- App Lifecycle Events ---

// This method will be called when Electron has finished initialization
// and is ready to create browser windows.
app.whenReady().then(createWindow);

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', () => {
    // Modified: Ensure quit even on macOS if explicitly closed
    app.quit();
    // killPythonProcess(); // Handled by will-quit
});

app.on('activate', () => {
    // On macOS it's common to re-create a window when the dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// Ensure Python process is killed when the Electron app quits
app.on('will-quit', () => {
    // Ensure Python process is killed before the app finally quits
    killPythonProcess();
});

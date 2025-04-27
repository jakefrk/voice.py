const { contextBridge, ipcRenderer } = require('electron');

// Expose specific IPC functions to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
    // Function to send messages from renderer to main
    send: (channel, data) => {
        // Whitelist channels
        const validChannels = ['start-analysis', 'stop-analysis', 'quit-app'];
        if (validChannels.includes(channel)) {
            ipcRenderer.send(channel, data);
        }
    },
    // Function to receive messages from main to renderer
    on: (channel, func) => {
        const validChannels = ['python-data', 'python-error', 'python-status'];
        if (validChannels.includes(channel)) {
            // Deliberately strip event as it includes `sender`
            ipcRenderer.on(channel, (event, ...args) => func(...args));
        }
    }
});

console.log('Preload script loaded.');

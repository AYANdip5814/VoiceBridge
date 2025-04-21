const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'electron',
  {
    // Video capture
    startCapture: (sourceId) => ipcRenderer.invoke('start-capture', sourceId),
    getVideoSources: () => ipcRenderer.invoke('get-video-sources'),
    
    // Settings
    getSettings: () => ipcRenderer.invoke('get-settings'),
    saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
    
    // Window management
    minimize: () => ipcRenderer.send('minimize-window'),
    maximize: () => ipcRenderer.send('maximize-window'),
    close: () => ipcRenderer.send('close-window'),
    
    // Events
    onCaptureStarted: (callback) => ipcRenderer.on('capture-started', callback),
    onCaptureStopped: (callback) => ipcRenderer.on('capture-stopped', callback),
    onSettingsChanged: (callback) => ipcRenderer.on('settings-changed', callback),
    
    // Remove listeners
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
  }
); 
const { app, BrowserWindow, ipcMain, desktopCapturer, screen } = require('electron');
const path = require('path');
const Store = require('electron-store');
const store = new Store();

let mainWindow;
let isCapturing = false;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    icon: path.join(__dirname, '../frontend/public/favicon.ico')
  });

  // In development, load from localhost
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../frontend/build/index.html'));
  }

  // Handle window state
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Video capture handling
ipcMain.handle('start-capture', async (event, sourceId) => {
  if (isCapturing) return;
  
  try {
    const sources = await desktopCapturer.getSources({
      types: ['window', 'screen'],
      thumbnailSize: { width: 1280, height: 720 }
    });

    const source = sourceId 
      ? sources.find(s => s.id === sourceId)
      : sources[0];

    if (!source) throw new Error('No video source found');

    isCapturing = true;
    return {
      id: source.id,
      name: source.name,
      thumbnail: source.thumbnail.toDataURL()
    };
  } catch (error) {
    console.error('Capture error:', error);
    throw error;
  }
});

// Integration with video calling apps
ipcMain.handle('get-video-sources', async () => {
  try {
    const sources = await desktopCapturer.getSources({
      types: ['window'],
      thumbnailSize: { width: 320, height: 180 }
    });

    // Filter for common video calling apps
    const videoApps = sources.filter(source => {
      const name = source.name.toLowerCase();
      return name.includes('zoom') ||
             name.includes('meet') ||
             name.includes('teams') ||
             name.includes('skype') ||
             name.includes('discord');
    });

    return videoApps.map(source => ({
      id: source.id,
      name: source.name,
      thumbnail: source.thumbnail.toDataURL()
    }));
  } catch (error) {
    console.error('Error getting video sources:', error);
    throw error;
  }
});

// Settings management
ipcMain.handle('get-settings', () => {
  return store.get('settings') || {
    language: 'asl',
    quality: 1.0,
    autoStart: false,
    hotkey: 'CommandOrControl+Shift+V'
  };
});

ipcMain.handle('save-settings', (event, settings) => {
  store.set('settings', settings);
  return true;
});

// App lifecycle
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Global shortcut registration
const { globalShortcut } = require('electron');

app.whenReady().then(() => {
  const settings = store.get('settings') || {};
  if (settings.hotkey) {
    globalShortcut.register(settings.hotkey, () => {
      if (mainWindow) {
        mainWindow.show();
        mainWindow.focus();
      }
    });
  }
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
}); 
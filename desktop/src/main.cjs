const { app, BrowserWindow, ipcMain, dialog, Menu, session, desktopCapturer, systemPreferences } = require('electron');
const path = require('path');
const fs = require('fs');
const net = require('net');
const { spawn } = require('child_process');

const DEFAULT_CONFIG = {
  sshHost: '',
  sshUser: 'ubuntu',
  sshPort: 22,
  identityFile: '',
  localPort: 18080,
  remoteHost: '127.0.0.1',
  remotePort: 8080,
  extraArgs: ''
};

let mainWindow = null;
let tunnelProcess = null;
let tunnelStatus = {
  connected: false,
  message: 'Disconnected',
  localUrl: ''
};

function configPath() {
  return path.join(app.getPath('userData'), 'connection-config.json');
}

function loadConfig() {
  try {
    const raw = fs.readFileSync(configPath(), 'utf8');
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

function saveConfig(config) {
  const merged = { ...DEFAULT_CONFIG, ...config };
  fs.writeFileSync(configPath(), JSON.stringify(merged, null, 2), 'utf8');
  return merged;
}

async function exportConfigToFile(config) {
  const merged = saveConfig(config);
  const result = await dialog.showSaveDialog({
    title: 'Save SSH Tunnel Config',
    defaultPath: 'scriberr-ssh-config.json',
    filters: [{ name: 'JSON', extensions: ['json'] }]
  });

  if (result.canceled || !result.filePath) {
    return { saved: false, path: '' };
  }

  fs.writeFileSync(result.filePath, JSON.stringify(merged, null, 2), 'utf8');
  return { saved: true, path: result.filePath };
}

async function importConfigFromFile() {
  const result = await dialog.showOpenDialog({
    title: 'Load SSH Tunnel Config',
    properties: ['openFile'],
    filters: [{ name: 'JSON', extensions: ['json'] }]
  });

  if (result.canceled || !result.filePaths.length) {
    return { loaded: false, path: '', config: loadConfig() };
  }

  const selectedPath = result.filePaths[0];
  const raw = fs.readFileSync(selectedPath, 'utf8');
  const parsed = JSON.parse(raw);

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Invalid config JSON format.');
  }

  const merged = saveConfig(parsed);
  return { loaded: true, path: selectedPath, config: merged };
}

function emitStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('tunnel:status', tunnelStatus);
  }
}

function setTunnelStatus(partial) {
  tunnelStatus = { ...tunnelStatus, ...partial };
  emitStatus();
}

function splitExtraArgs(extraArgs) {
  if (!extraArgs || typeof extraArgs !== 'string') {
    return [];
  }
  return extraArgs
    .split(' ')
    .map((s) => s.trim())
    .filter(Boolean);
}

function isLocalPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close(() => resolve(true));
    });
    server.listen(port, '127.0.0.1');
  });
}

function waitForTunnelReady(proc, port, getLastError, timeoutMs = 15000) {
  const start = Date.now();

  return new Promise((resolve, reject) => {
    let done = false;

    const finish = (err) => {
      if (done) {
        return;
      }
      done = true;
      proc.off('exit', onExit);
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    };

    const onExit = (code, signal) => {
      const details = getLastError();
      const fallback = `SSH exited early (code=${code ?? 'n/a'}, signal=${signal ?? 'n/a'})`;
      finish(new Error(details || fallback));
    };

    const attempt = () => {
      if (done) {
        return;
      }

      const socket = net.createConnection({ host: '127.0.0.1', port });

      socket.on('connect', () => {
        socket.end();
        finish();
      });

      socket.on('error', () => {
        socket.destroy();
        if (Date.now() - start >= timeoutMs) {
          const details = getLastError();
          finish(new Error(details || `Timed out waiting for local port ${port}`));
          return;
        }
        setTimeout(attempt, 250);
      });
    };

    proc.once('exit', onExit);
    attempt();
  });
}

function stopTunnelInternal() {
  if (tunnelProcess && !tunnelProcess.killed) {
    tunnelProcess.kill('SIGTERM');
  }
  tunnelProcess = null;
  setTunnelStatus({ connected: false, message: 'Disconnected', localUrl: '' });
}

async function startTunnel(config) {
  if (tunnelProcess && tunnelStatus.connected) {
    return tunnelStatus;
  }
  if (tunnelProcess) {
    stopTunnelInternal();
  }

  const effective = saveConfig(config);

  if (!effective.sshHost || !effective.sshUser) {
    throw new Error('sshHost and sshUser are required.');
  }
  const localPortFree = await isLocalPortAvailable(effective.localPort);
  if (!localPortFree) {
    if (effective.localPort === 8080) {
      const fallbackPort = 18080;
      const fallbackFree = await isLocalPortAvailable(fallbackPort);
      if (fallbackFree) {
        effective.localPort = fallbackPort;
        saveConfig(effective);
        setTunnelStatus({ message: 'Local port 8080 is in use, switched to 18080.' });
      } else {
        throw new Error('Local ports 8080 and 18080 are both in use. Pick another local port.');
      }
    } else {
      throw new Error(`Local port ${effective.localPort} is already in use.`);
    }
  }

  const args = [
    '-N',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-L', `${effective.localPort}:${effective.remoteHost}:${effective.remotePort}`,
    '-p', String(effective.sshPort)
  ];

  if (effective.identityFile) {
    args.push('-i', effective.identityFile);
  }

  args.push(...splitExtraArgs(effective.extraArgs));
  args.push(`${effective.sshUser}@${effective.sshHost}`);

  setTunnelStatus({ connected: false, message: 'Connecting...', localUrl: '' });

  tunnelProcess = spawn('ssh', args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let lastSshError = '';

  tunnelProcess.stdout.on('data', (data) => {
    const text = data.toString().trim();
    if (text) {
      setTunnelStatus({ message: text });
    }
  });

  tunnelProcess.stderr.on('data', (data) => {
    const text = data.toString().trim();
    if (text) {
      lastSshError = text;
      setTunnelStatus({ message: text });
    }
  });

  tunnelProcess.on('exit', (code, signal) => {
    tunnelProcess = null;
    setTunnelStatus({
      connected: false,
      localUrl: '',
      message: `Disconnected (code=${code ?? 'n/a'}, signal=${signal ?? 'n/a'})`
    });
  });

  try {
    await waitForTunnelReady(tunnelProcess, effective.localPort, () => lastSshError, 15000);
  } catch (err) {
    stopTunnelInternal();
    throw err;
  }

  const localUrl = `http://127.0.0.1:${effective.localPort}`;
  setTunnelStatus({ connected: true, localUrl, message: `Connected: ${localUrl}` });
  return tunnelStatus;
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    title: 'Scriberr Desktop',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webviewTag: true
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function setupPermissionHandlers() {
  const ses = session.defaultSession;
  if (!ses) {
    return;
  }

  const isMediaPermission = (permission) => {
    return permission === 'media' ||
      permission === 'microphone' ||
      permission === 'camera' ||
      permission === 'display-capture' ||
      permission === 'audioCapture' ||
      permission === 'videoCapture';
  };

  // In packaged desktop builds, webview media requests can be attributed to the file:// embedder.
  // Allow all media permissions here and rely on macOS TCC prompts for final user consent.
  ses.setPermissionCheckHandler((_webContents, permission) => {
    return isMediaPermission(permission);
  });

  ses.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(isMediaPermission(permission));
  });

  if (typeof ses.setDisplayMediaRequestHandler === 'function') {
    ses.setDisplayMediaRequestHandler(async (_request, callback) => {
      try {
        const sources = await desktopCapturer.getSources({ types: ['screen', 'window'] });
        callback({
          video: sources[0] || null,
          audio: 'loopback'
        });
      } catch (error) {
        console.error('Display media request handler failed:', error);
        callback({ video: null, audio: null });
      }
    }, { useSystemPicker: true });
  }
}

async function ensureMacPermissions() {
  if (process.platform !== 'darwin') {
    return;
  }

  try {
    const micStatus = systemPreferences.getMediaAccessStatus('microphone');
    if (micStatus !== 'granted') {
      await systemPreferences.askForMediaAccess('microphone');
    }
  } catch (error) {
    console.error('Failed to ensure microphone permission:', error);
  }
}

function setupMenu() {
  const template = [
    {
      label: 'Scriberr Desktop',
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        {
          label: 'Disconnect Tunnel',
          click: () => stopTunnelInternal()
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'pasteAndMatchStyle' },
        { role: 'delete' },
        { role: 'selectAll' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

ipcMain.handle('config:get', () => loadConfig());
ipcMain.handle('config:save', (_event, config) => saveConfig(config || {}));
ipcMain.handle('config:export', (_event, config) => exportConfigToFile(config || {}));
ipcMain.handle('config:import', () => importConfigFromFile());
ipcMain.handle('tunnel:start', async (_event, config) => startTunnel(config || {}));
ipcMain.handle('tunnel:stop', () => {
  stopTunnelInternal();
  return tunnelStatus;
});
ipcMain.handle('tunnel:status', () => tunnelStatus);
ipcMain.handle('session:clearSiteData', async (_event, originUrl) => {
  const ses = session.defaultSession;
  if (!ses) {
    return false;
  }

  try {
    await ses.clearCache();
    if (originUrl && typeof originUrl === 'string') {
      await ses.clearStorageData({
        origin: originUrl,
        storages: ['serviceworkers', 'cachestorage', 'indexdb', 'localstorage']
      });
    } else {
      await ses.clearStorageData({
        storages: ['serviceworkers', 'cachestorage', 'indexdb', 'localstorage']
      });
    }
    return true;
  } catch (err) {
    console.error('Failed to clear site data:', err);
    return false;
  }
});
ipcMain.handle('config:chooseIdentityFile', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Select SSH Private Key',
    properties: ['openFile'],
    filters: [{ name: 'SSH Keys', extensions: ['pem', 'key', 'ppk', '*'] }]
  });

  if (result.canceled || !result.filePaths.length) {
    return '';
  }
  return result.filePaths[0];
});

app.whenReady().then(async () => {
  await ensureMacPermissions();
  setupPermissionHandlers();
  setupMenu();
  createMainWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('before-quit', () => {
  stopTunnelInternal();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

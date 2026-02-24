const ids = [
  'sshHost', 'sshUser', 'sshPort', 'identityFile',
  'localPort', 'remoteHost', 'remotePort', 'extraArgs'
];

const fields = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const backBtn = document.getElementById('backBtn');
const saveConfigBtn = document.getElementById('saveConfigBtn');
const loadConfigBtn = document.getElementById('loadConfigBtn');
const reloadBtn = document.getElementById('reloadBtn');
const chooseIdentityBtn = document.getElementById('chooseIdentity');
const statusEl = document.getElementById('status');
const viewer = document.getElementById('viewer');
backBtn.disabled = true;

let currentUrl = '';
let appView = null;

function readForm() {
  return {
    sshHost: fields.sshHost.value.trim(),
    sshUser: fields.sshUser.value.trim(),
    sshPort: Number(fields.sshPort.value || 22),
    identityFile: fields.identityFile.value.trim(),
    localPort: Number(fields.localPort.value || 18080),
    remoteHost: fields.remoteHost.value.trim() || '127.0.0.1',
    remotePort: Number(fields.remotePort.value || 8080),
    extraArgs: fields.extraArgs.value.trim()
  };
}

function applyForm(config) {
  fields.sshHost.value = config.sshHost || '';
  fields.sshUser.value = config.sshUser || 'ubuntu';
  fields.sshPort.value = String(config.sshPort || 22);
  fields.identityFile.value = config.identityFile || '';
  fields.localPort.value = String(config.localPort || 18080);
  fields.remoteHost.value = config.remoteHost || '127.0.0.1';
  fields.remotePort.value = String(config.remotePort || 8080);
  fields.extraArgs.value = config.extraArgs || '';
}

function setStatus(status) {
  const text = status?.message || 'Disconnected';
  statusEl.textContent = text;
  statusEl.classList.remove('ok', 'err');
  if (status?.connected) {
    statusEl.classList.add('ok');
  } else if (text.toLowerCase().includes('error') || text.toLowerCase().includes('timed out')) {
    statusEl.classList.add('err');
  }

  disconnectBtn.disabled = !status?.connected;
  connectBtn.disabled = Boolean(status?.connected);
}

function appendCacheBust(url) {
  try {
    const parsed = new URL(url);
    parsed.searchParams.set('_t', Date.now().toString());
    return parsed.toString();
  } catch {
    return url;
  }
}

function updateBackButtonState() {
  if (!appView) {
    backBtn.disabled = true;
    return;
  }
  try {
    backBtn.disabled = !appView.canGoBack();
  } catch {
    backBtn.disabled = true;
  }
}

function renderViewer(url, options = {}) {
  const { forceFresh = false } = options;
  viewer.innerHTML = '';
  const webview = document.createElement('webview');
  webview.src = forceFresh ? appendCacheBust(url) : url;
  webview.allow = 'microphone; camera; display-capture; autoplay; clipboard-read; clipboard-write';
  webview.setAttribute('allow', 'microphone; camera; display-capture; autoplay; clipboard-read; clipboard-write');
  webview.setAttribute('allowpopups', 'false');
  webview.style.width = '100%';
  webview.style.height = '100%';
  webview.style.border = '0';

  webview.addEventListener('did-finish-load', () => {
    updateBackButtonState();
  });
  webview.addEventListener('did-navigate', () => {
    updateBackButtonState();
  });
  webview.addEventListener('did-navigate-in-page', () => {
    updateBackButtonState();
  });
  webview.addEventListener('did-fail-load', (event) => {
    if (event.errorCode === -3) {
      // Ignore ERR_ABORTED from internal redirects.
      return;
    }
    statusEl.textContent = `Error: ${event.errorDescription || 'Failed to load app'}`;
    statusEl.classList.add('err');
  });

  viewer.appendChild(webview);
  appView = webview;
  backBtn.disabled = true;
}

function clearViewer() {
  currentUrl = '';
  appView = null;
  viewer.innerHTML = '<div class="placeholder">连接后将在此显示 Scriberr 页面。</div>';
  backBtn.disabled = true;
}

async function loadAppUrl(url, { forceFresh = true } = {}) {
  currentUrl = url;
  if (forceFresh) {
    await window.desktopBridge.clearSiteData(url);
  }
  renderViewer(url, { forceFresh });
}

async function init() {
  const config = await window.desktopBridge.getConfig();
  applyForm(config);

  const status = await window.desktopBridge.getTunnelStatus();
  setStatus(status);

  if (status?.connected && status.localUrl) {
    await loadAppUrl(status.localUrl, { forceFresh: true });
  }
}

connectBtn.addEventListener('click', async () => {
  connectBtn.disabled = true;
  statusEl.textContent = 'Connecting...';
  statusEl.classList.remove('ok', 'err');

  try {
    const config = readForm();
    await window.desktopBridge.saveConfig(config);
    const status = await window.desktopBridge.startTunnel(config);
    setStatus(status);
    if (status?.connected && status.localUrl) {
      await loadAppUrl(status.localUrl, { forceFresh: true });
    }
  } catch (err) {
    statusEl.textContent = `Error: ${err?.message || 'Failed to connect'}`;
    statusEl.classList.add('err');
    connectBtn.disabled = false;
  }
});

disconnectBtn.addEventListener('click', async () => {
  const status = await window.desktopBridge.stopTunnel();
  setStatus(status);
  clearViewer();
});

reloadBtn.addEventListener('click', () => {
  if (currentUrl) {
    if (appView) {
      appView.reloadIgnoringCache();
      return;
    }
    renderViewer(currentUrl, { forceFresh: true });
  }
});

backBtn.addEventListener('click', async () => {
  if (!appView) {
    return;
  }
  try {
    if (appView.canGoBack()) {
      appView.goBack();
    }
    updateBackButtonState();
  } catch (err) {
    statusEl.textContent = `Error: ${err?.message || 'Failed to navigate back'}`;
    statusEl.classList.add('err');
  }
});

chooseIdentityBtn.addEventListener('click', async () => {
  const selected = await window.desktopBridge.chooseIdentityFile();
  if (selected) {
    fields.identityFile.value = selected;
  }
});

saveConfigBtn.addEventListener('click', async () => {
  try {
    const config = readForm();
    await window.desktopBridge.saveConfig(config);
    const result = await window.desktopBridge.exportConfig(config);
    if (result?.saved) {
      statusEl.textContent = `Config saved: ${result.path}`;
      statusEl.classList.remove('err');
    }
  } catch (err) {
    statusEl.textContent = `Error: ${err?.message || 'Failed to save config'}`;
    statusEl.classList.add('err');
  }
});

loadConfigBtn.addEventListener('click', async () => {
  try {
    const result = await window.desktopBridge.importConfig();
    if (result?.loaded) {
      applyForm(result.config || {});
      statusEl.textContent = `Config loaded: ${result.path}`;
      statusEl.classList.remove('err');
      return;
    }
    statusEl.textContent = 'Load cancelled';
    statusEl.classList.remove('err');
  } catch (err) {
    statusEl.textContent = `Error: ${err?.message || 'Failed to load config'}`;
    statusEl.classList.add('err');
  }
});

window.desktopBridge.onTunnelStatus((status) => {
  setStatus(status);
  if (!status?.connected) {
    if (!status?.message || status.message.startsWith('Disconnected')) {
      clearViewer();
    }
    return;
  }

  if (status.localUrl && status.localUrl !== currentUrl) {
    loadAppUrl(status.localUrl, { forceFresh: true }).catch((err) => {
      statusEl.textContent = `Error: ${err?.message || 'Failed to load app'}`;
      statusEl.classList.add('err');
    });
  }
});

init().catch((err) => {
  statusEl.textContent = `Error: ${err?.message || 'Init failed'}`;
  statusEl.classList.add('err');
});

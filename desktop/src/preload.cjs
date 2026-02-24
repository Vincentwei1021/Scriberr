const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopBridge', {
  getConfig: () => ipcRenderer.invoke('config:get'),
  saveConfig: (config) => ipcRenderer.invoke('config:save', config),
  exportConfig: (config) => ipcRenderer.invoke('config:export', config),
  importConfig: () => ipcRenderer.invoke('config:import'),
  chooseIdentityFile: () => ipcRenderer.invoke('config:chooseIdentityFile'),
  startTunnel: (config) => ipcRenderer.invoke('tunnel:start', config),
  stopTunnel: () => ipcRenderer.invoke('tunnel:stop'),
  getTunnelStatus: () => ipcRenderer.invoke('tunnel:status'),
  clearSiteData: (originUrl) => ipcRenderer.invoke('session:clearSiteData', originUrl),
  onTunnelStatus: (callback) => {
    const handler = (_event, status) => callback(status);
    ipcRenderer.on('tunnel:status', handler);
    return () => ipcRenderer.removeListener('tunnel:status', handler);
  }
});

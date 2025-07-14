const { ipcRenderer } = require("electron");

window.addEventListener("DOMContentLoaded", () => {});

// Expose window control methods to the renderer process
window.electronAPI = {
  minimizeWindow: () => ipcRenderer.invoke("window-minimize"),
  maximizeWindow: () => ipcRenderer.invoke("window-maximize"),
  closeWindow: () => ipcRenderer.invoke("window-close"),
  isMaximized: () => ipcRenderer.invoke("window-is-maximized"),
  onWindowMaximized: (callback) => {
    ipcRenderer.on("window-maximized", (event, isMaximized) =>
      callback(isMaximized),
    );
  },
};

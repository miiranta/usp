import { app, BrowserWindow, ipcMain } from "electron";
import { format } from "url";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

// Run the Express server
import expressApp from "./express/express.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    minWidth: 1200,
    minHeight: 600,
    frame: false,
    titleBarStyle: "hidden",
    icon: path.join(__dirname, "/assets/icon.png"),

    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, "preloader.js"),
    },
  });

  mainWindow.loadURL(
    format({
      pathname: path.join(__dirname, "../dist/plant-growth/browser/index.html"),
      protocol: "file:",
      slashes: true,
    }),
  );

  mainWindow.removeMenu();

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Listen for maximize/unmaximize events to update UI
  mainWindow.on("maximize", () => {
    mainWindow.webContents.send("window-maximized", true);
  });

  mainWindow.on("unmaximize", () => {
    mainWindow.webContents.send("window-maximized", false);
  });

  console.log("Electron is up.");
}

// Window control handlers
ipcMain.handle("window-minimize", () => {
  if (mainWindow) {
    mainWindow.minimize();
  }
});

ipcMain.handle("window-maximize", () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.restore();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.handle("window-close", () => {
  if (mainWindow) {
    mainWindow.close();
  }
});

ipcMain.handle("window-is-maximized", () => {
  if (mainWindow) {
    return mainWindow.isMaximized();
  }
  return false;
});

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (mainWindow === null) createWindow();
});

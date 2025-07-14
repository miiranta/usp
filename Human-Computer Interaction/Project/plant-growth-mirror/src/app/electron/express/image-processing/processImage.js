import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";
import fetch from "node-fetch";
import * as tar from "tar";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORTABLE_DIR = path.join(__dirname, 'python-portable');
const PYTHON_VERSION = '3.11.12';
const RELEASE_DATE = '20250409';
const SETUP_COMPLETE_MARKER = path.join(PORTABLE_DIR, '.setup_complete');

async function isOnline() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
    const response = await fetch('https://httpbin.org/status/200', {
      method: 'HEAD',
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    return false;
  }
}

async function downloadFile(url, destPath) {
  console.log(`Downloading from: ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url}: ${res.statusText}`);
  
  return new Promise((resolve, reject) => {
    const fileStream = fs.createWriteStream(destPath);
    res.body.pipe(fileStream);
    res.body.on('error', reject);
    fileStream.on('finish', resolve);
    fileStream.on('error', reject);
  });
}

function markSetupComplete() {
  try {
    const metadata = {
      timestamp: new Date().toISOString(),
      pythonVersion: PYTHON_VERSION,
      releaseDate: RELEASE_DATE,
      platform: process.platform
    };
    fs.writeFileSync(SETUP_COMPLETE_MARKER, JSON.stringify(metadata, null, 2));
    console.log('Setup marked as complete');
  } catch (error) {
    console.warn('Failed to create setup marker:', error.message);
  }
}

function isSetupComplete() {
  if (!fs.existsSync(SETUP_COMPLETE_MARKER)) {
    return false;
  }
  
  try {
    const metadata = JSON.parse(fs.readFileSync(SETUP_COMPLETE_MARKER, 'utf8'));
    // Verify the setup is for the current platform and version
    return metadata.pythonVersion === PYTHON_VERSION && 
           metadata.platform === process.platform;
  } catch (error) {
    console.warn('Setup marker corrupted, will re-setup');
    return false;
  }
}

async function setupPortablePython() {
  if (fs.existsSync(PORTABLE_DIR)) {
    console.log('Removing existing portable Python directory...');
    fs.rmSync(PORTABLE_DIR, { recursive: true, force: true });
  }
  
  console.log('Creating portable Python directory...');
  fs.mkdirSync(PORTABLE_DIR, { recursive: true });

  try {
    let tarUrl;
    
    if (process.platform === 'win32') {
      tarUrl = `https://github.com/astral-sh/python-build-standalone/releases/download/${RELEASE_DATE}/cpython-${PYTHON_VERSION}+${RELEASE_DATE}-x86_64-pc-windows-msvc-install_only.tar.gz`;
    } else if (process.platform === 'linux') {
      tarUrl = `https://github.com/astral-sh/python-build-standalone/releases/download/${RELEASE_DATE}/cpython-${PYTHON_VERSION}+${RELEASE_DATE}-x86_64-unknown-linux-gnu-install_only.tar.gz`;
    } else {
      throw new Error(`Unsupported platform: ${process.platform}. This setup only supports Windows and Linux.`);
    }
    
    const tarPath = path.join(PORTABLE_DIR, 'python.tar.gz');
    console.log(`Downloading Python standalone for ${process.platform}...`);
    await downloadFile(tarUrl, tarPath);
    
    console.log('Extracting archive...');
    await tar.x({ 
      file: tarPath, 
      C: PORTABLE_DIR, 
      strip: 1,
      onwarn: (message, data) => console.warn('TAR warning:', message)
    });
    
    // Clean up downloaded archive
    fs.unlinkSync(tarPath);
    
    // Mark setup as complete
    markSetupComplete();
    console.log('Portable Python setup completed');
    
  } catch (error) {
    console.error('Failed to setup portable Python:', error.message);
    // Clean up on failure
    if (fs.existsSync(PORTABLE_DIR)) {
      fs.rmSync(PORTABLE_DIR, { recursive: true, force: true });
    }
    throw error;
  }
}

function getPortablePythonPath() {
  if (process.platform === 'win32') {
    const windowsPaths = [
      path.join(PORTABLE_DIR, 'python.exe'),
      path.join(PORTABLE_DIR, 'bin', 'python.exe'),
      path.join(PORTABLE_DIR, 'install', 'python.exe'),
      path.join(PORTABLE_DIR, 'install', 'bin', 'python.exe')
    ];
    
    for (const pythonPath of windowsPaths) {
      if (fs.existsSync(pythonPath)) {
        console.log(`Found portable Python at: ${pythonPath}`);
        return pythonPath;
      }
    }
  } else {
    const linuxPaths = [
      path.join(PORTABLE_DIR, 'bin', 'python3.11'),
      path.join(PORTABLE_DIR, 'bin', 'python3'),
      path.join(PORTABLE_DIR, 'bin', 'python'),
      path.join(PORTABLE_DIR, 'install', 'bin', 'python3.11'),
      path.join(PORTABLE_DIR, 'install', 'bin', 'python3')
    ];
    
    for (const pythonPath of linuxPaths) {
      if (fs.existsSync(pythonPath)) {
        console.log(`Found portable Python at: ${pythonPath}`);
        return pythonPath;
      }
    }
  }
  
  return null;
}

async function getOrSetupPortablePython() {
  // Check if setup is already complete
  if (isSetupComplete()) {
    const pythonPath = getPortablePythonPath();
    if (pythonPath) {
      console.log('Using existing portable Python installation');
      return pythonPath;
    } else {
      console.log('Setup marked complete but Python executable not found, will re-setup');
    }
  }
  
  // Check if we're online for initial setup
  const online = await isOnline();
  if (!online && !fs.existsSync(PORTABLE_DIR)) {
    throw new Error('No internet connection available and no cached Python installation found. Please connect to the internet for initial setup.');
  }
  
  if (!online && !isSetupComplete()) {
    console.log('Offline mode: Attempting to use existing Python installation...');
    const pythonPath = getPortablePythonPath();
    if (pythonPath) {
      console.log('Found existing Python installation, proceeding offline');
      return pythonPath;
    } else {
      throw new Error('No internet connection and no valid Python installation found. Please connect to the internet for initial setup.');
    }
  }
  
  // Setup or re-setup Python
  await setupPortablePython();
  const pythonPath = getPortablePythonPath();
  
  if (!pythonPath) {
    throw new Error('Failed to locate Python executable in portable installation');
  }
  
  return pythonPath;
}

function isVirtualEnvironmentComplete() {
  const venvDir = path.join(__dirname, '.venv');
  const venvPython = process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');
  
  const venvMarker = path.join(venvDir, '.venv_complete');
  
  return fs.existsSync(venvPython) && fs.existsSync(venvMarker);
}

function markVenvComplete() {
  const venvDir = path.join(__dirname, '.venv');
  const venvMarker = path.join(venvDir, '.venv_complete');
  
  try {
    const metadata = {
      timestamp: new Date().toISOString(),
      platform: process.platform
    };
    fs.writeFileSync(venvMarker, JSON.stringify(metadata, null, 2));
    console.log('Virtual environment marked as complete');
  } catch (error) {
    console.warn('Failed to create venv marker:', error.message);
  }
}

async function setupVirtualEnvironment() {
  const pythonCmd = await getOrSetupPortablePython();
  console.log('Using Python:', pythonCmd);

  const venvDir = path.join(__dirname, '.venv');
  const venvPython = process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');

  // Check if virtual environment is already complete
  if (isVirtualEnvironmentComplete()) {
    console.log('Using existing virtual environment');
    return venvPython;
  }

  // Check internet connectivity for package installation
  const online = await isOnline();
  if (!online) {
    if (fs.existsSync(venvPython)) {
      console.log('Offline mode: Using existing virtual environment (packages may not be up to date)');
      return venvPython;
    } else {
      throw new Error('No internet connection available and no virtual environment found. Please connect to the internet for initial setup.');
    }
  }

  console.log('Setting up virtual environment...');
  
  // Remove existing venv if it exists but is incomplete
  if (fs.existsSync(venvDir)) {
    console.log('Removing incomplete virtual environment...');
    fs.rmSync(venvDir, { recursive: true, force: true });
  }
  
  console.log('Creating virtual environment...');
  const { execFileSync } = require('child_process');
  execFileSync(pythonCmd, ['-m', 'venv', '--upgrade-deps', venvDir], { 
    stdio: 'inherit',
    timeout: 60000 // 1 minute timeout
  });

  const execOptions = { 
    stdio: 'inherit',
    timeout: 300000 // 5 minute timeout for package installations
  };

  console.log('Upgrading pip, setuptools, wheel...');
  execFileSync(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], execOptions);

  console.log('Installing numpy and opencv-python...');
  execFileSync(venvPython, ['-m', 'pip', 'install', '--only-binary=:all:', 'numpy', 'opencv-python'], execOptions);

  const requirementsPath = path.join(__dirname, 'requirements.txt');
  if (fs.existsSync(requirementsPath)) {
    console.log('Installing requirements.txt...');
    execFileSync(venvPython, ['-m', 'pip', 'install', '--no-cache-dir', '--timeout', '300', '-r', requirementsPath], execOptions);
  } else {
    console.warn('requirements.txt not found, skipping...');
  }

  // Mark virtual environment as complete
  markVenvComplete();

  return venvPython;
}

export async function testPythonEnvironment() {
  try {
    const py = await setupVirtualEnvironment();
    const testScriptPath = path.join(__dirname, 'test_setup.py');
    
    if (!fs.existsSync(testScriptPath)) {
      return { success: false, error: 'test_setup.py not found' };
    }
    
    const { execFileSync } = require('child_process');
    const output = execFileSync(py, [testScriptPath], { 
      encoding: 'utf-8',
      timeout: 30000 // 30 second timeout
    });
    console.log(output);
    return { success: true, output };
  } catch (error) {
    console.error('Python environment test failed:', error.message);
    return { success: false, error: error.message };
  }
}

export async function checkOfflineCapability() {
  try {
    const setupComplete = isSetupComplete();
    const venvComplete = isVirtualEnvironmentComplete();
    const online = await isOnline();
    
    return {
      setupComplete,
      venvComplete,
      online,
      canRunOffline: setupComplete && venvComplete,
      pythonPath: setupComplete ? getPortablePythonPath() : null
    };
  } catch (error) {
    return {
      setupComplete: false,
      venvComplete: false,
      online: false,
      canRunOffline: false,
      pythonPath: null,
      error: error.message
    };
  }
}

export default async function processImage(id, dataBase64, granularity, threshold) {
  const errorResult = { 
    dataProcessedBase64: "", 
    result: { height: 0, width: 0, area: 0 },
    error: "Processing failed"
  };
  
  try {
    // Validate inputs
    if (!id || !dataBase64) {
      throw new Error('Missing required parameters: id and dataBase64');
    }
    
    const py = await setupVirtualEnvironment();
    const tempDir = path.join(__dirname, 'temp');
    fs.mkdirSync(tempDir, { recursive: true });

    const jsonPath = path.join(tempDir, `${id}.json`);
    const inputData = {
      id,
      dataBase64,
      granularity: Math.max(Number(granularity) || 20, 2),
      threshold: Math.max(Number(threshold) || 0.05, 0)
    };
    
    fs.writeFileSync(jsonPath, JSON.stringify(inputData));

    const processScriptPath = path.join(__dirname, 'processImage.py');
    if (!fs.existsSync(processScriptPath)) {
      throw new Error('processImage.py not found');
    }

    const { execFileSync } = require('child_process');
    const output = execFileSync(py, [processScriptPath, id], { 
      encoding: 'utf-8', 
      maxBuffer: 50 * 1024 * 1024, // 50MB buffer
      timeout: 120000 // 2 minute timeout
    });
    
    // Clean up temp file
    try {
      fs.unlinkSync(jsonPath);
    } catch (cleanupError) {
      console.warn('Failed to clean up temp file:', cleanupError.message);
    }

    if (!output || output.trim() === '') {
      throw new Error('No output received from Python script');
    }

    const result = JSON.parse(output);
    return result.dataProcessedBase64 ? result : errorResult;
    
  } catch (error) {
    console.error('Image processing error:', error.message);
    return { ...errorResult, error: error.message };
  }
}
const SHOW_INFO = false;
const SHOW_SUCCESS = true;
const SHOW_ERROR = true;
const SHOW_WARN = true;

const colors = {
  reset: "\x1b[0m",
  white: "\x1b[37m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

const log = {
  info: (module, message) => {
    if (!SHOW_INFO) return;
    console.log(`${colors.white}[${module}] ${message}${colors.reset}`);
  },
  success: (module, message) => {
    if (!SHOW_SUCCESS) return;
    console.log(`${colors.green}[${module}] ${message}${colors.reset}`);
  },
  error: (module, message) => {
    if (!SHOW_ERROR) return;
    console.log(`${colors.red}[${module}] ${message}${colors.reset}`);
  },
  warn: (module, message) => {
    if (!SHOW_WARN) return;
    console.log(`${colors.yellow}[${module}] ${message}${colors.reset}`);
  },
};

module.exports = { colors, log };

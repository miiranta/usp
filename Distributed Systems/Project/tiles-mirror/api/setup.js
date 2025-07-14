const fs = require("fs");
const path = require("path");

const dotenvPath = path.join(__dirname, "environments", ".env");
require("dotenv").config({ path: dotenvPath });

let config = {
  BASE_URL: "localhost",
  PORT: 3000,
};

function setAngularEnv() {
  const angularEnvDir = path.join(__dirname, "..", "app", "environments");
  const angularEnvFile = path.join(angularEnvDir, "environment.ts");

  if (process.env.DOMAIN) config.BASE_URL = process.env.DOMAIN;
  if (process.env.PORT) config.PORT = process.env.PORT;

  if (process.env.SSL === "true") {
    config.BASE_URL = `https://${config.BASE_URL}`;
    config.PORT = 443;
  } else {
    config.BASE_URL = `http://${config.BASE_URL}`;
  }

  if (!fs.existsSync(angularEnvDir)) {
    fs.mkdirSync(angularEnvDir, { recursive: true });
  }

  generateEnvironmentFile(angularEnvFile, config);
}

function generateEnvironmentFile(filePath, config) {
  const configEntries = Object.entries(config)
    .map(([key, value]) => `\n  ${key}: ${JSON.stringify(value)}`)
    .join(",");

  const content = `export const environment = {${configEntries}\n};`;

  try {
    fs.writeFileSync(filePath, content, "utf8");
  } catch (error) {
    console.error(`Erro ao escrever environment.ts:`, error.message);
  }
}

setAngularEnv();

const express = require("express");
const path = require("path");
const cors = require("cors");
const Greenlock = require("greenlock-express");
const { Server } = require("socket.io");
const { Database } = require("./infrastructure/database");
const { AppManager } = require("./application/appManager");
const { log } = require("./utils/colorLogging");
const fs = require("fs");

const ANGULAR_FOLDER = path.join(__dirname, "../../app/dist/tiles/browser");

require("dotenv").config({
  path: path.join(__dirname, "../environments/.env"),
});
const EMAIL = process.env.EMAIL;
const DOMAIN = process.env.DOMAIN;

if (!EMAIL || !DOMAIN) {
  console.error("EMAIL e DOMAIN devem ser definidos no arquivo .env");
  process.exit(1);
}

const greenlockDir = path.join(__dirname, "../greenlock.d");
const greenlockConfigPath = path.join(greenlockDir, "config.json");

try {
  if (!fs.existsSync(greenlockDir)) {
    fs.mkdirSync(greenlockDir, { recursive: true });
  }

  if (!fs.existsSync(greenlockConfigPath)) {
    const config = {
      sites: [
        {
          subject: DOMAIN,
          altnames: [DOMAIN, `www.${DOMAIN}`],
        },
      ],
    };
    fs.writeFileSync(greenlockConfigPath, JSON.stringify(config, null, 2));
    log.info(
      "setup",
      `Configuração do Greenlock criada: ${greenlockConfigPath}`,
    );
  }
} catch (error) {
  log.error("setup", `Erro ao criar configuração: ${error.message}`);
}

const startServer = async () => {
  const database = new Database();
  await database.connect();

  const app = express();
  app.use(express.json());
  app.use(cors({ origin: "*" }));

  app.use(express.static(ANGULAR_FOLDER));

  const glx = Greenlock.init({
    packageRoot: path.join(__dirname, ".."),
    configDir: path.join(__dirname, "../greenlock.d"),
    maintainerEmail: EMAIL,
    cluster: false,
  }).ready(httpsWorker);

  function httpsWorker(glx) {
    const server = glx.httpsServer();

    const io = new Server(server, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"],
        credentials: true,
      },
      allowEIO3: true,
      transports: ["websocket", "polling"],
    });

    new AppManager(app, io, database);

    //GET /: Envia o frontend ao cliente
    app.get("*", (req, res) => {
      res.sendFile(path.join(ANGULAR_FOLDER, "/index.html"));
    });

    glx.serveApp(app);

    log.success(
      "server",
      "Servidor iniciado com SSL habilitado. PORTA: 443 e 80.",
    );
  }

  const gracefulShutdown = async () => {
    log.info("server", "Desligando servidor...");
    await database.disconnect();
    process.exit(0);
  };

  process.on("SIGINT", gracefulShutdown);
  process.on("SIGTERM", gracefulShutdown);
};

startServer().catch((err) => {
  log.error("server", "Erro ao iniciar o servidor:", err);
  process.exit(1);
});

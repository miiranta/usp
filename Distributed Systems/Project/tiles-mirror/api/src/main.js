const express = require("express");
const http = require("http");
const path = require("path");
const cors = require("cors");
const { Server } = require("socket.io");
const { Database } = require("./infrastructure/database");
const { AppManager } = require("./application/appManager");
const { log } = require("./utils/colorLogging");

const dotenvPath = path.join(__dirname, "../environments", ".env");
require("dotenv").config({ path: dotenvPath });

const PORT = process.env.PORT || 3000;
const ANGULAR_FOLDER = path.join(__dirname, "../../app/dist/tiles/browser");

const startServer = async () => {
  const database = new Database();
  await database.connect();

  const app = express();
  app.use(express.json());
  app.use(cors({ origin: "*" }));

  app.use(express.static(ANGULAR_FOLDER));

  const server = http.createServer(app);
  const io = new Server(server, {
    cors: {
      origin: "*",
    },
  });

  new AppManager(app, io, database);

  //GET /: Envia o frontend ao cliente
  app.get("*", (req, res) => {
    res.sendFile(path.join(ANGULAR_FOLDER, "/index.html"));
  });

  server.listen(PORT, () => {
    log.success("server", `Servidor iniciado. PORTA: ${PORT}`);
  });

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

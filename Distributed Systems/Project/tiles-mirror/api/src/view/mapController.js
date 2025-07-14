const express = require("express");
const { Tile } = require("../domain/models/tileModel");
const { COLORS } = require("../domain/enums/colors");
const { log } = require("../utils/colorLogging");

class MapController {
  constructor(app, io, mapManager, tokenManager) {
    this.app = app;
    this.io = io;
    this.mapManager = mapManager;
    this.tokenManager = tokenManager;
    this.setupRoutes();
    this.setupWebSocket();
  }

  setupRoutes() {
    const router = express.Router();

    // GET /map/:x/:y/:range: Recupera blocos do mapa
    router.get("/map/:x/:y/:range", async (req, res) => {
      try {
        if (!req.params.x || !req.params.y || !req.params.range) {
          return res.status(400).json({ error: "Parâmetros faltando" });
        }

        const x = parseInt(req.params.x);
        const y = parseInt(req.params.y);
        const range = parseInt(req.params.range);

        if (range > 100) {
          return res
            .status(400)
            .json({ error: "Valor do range muito alto (>100)" });
        }
        if (range < 0) {
          return res
            .status(400)
            .json({ error: "Valor do range muito baixo (<0)" });
        }
        if (isNaN(x) || isNaN(y) || isNaN(range)) {
          return res
            .status(400)
            .json({ error: "Coordenadas ou valor de range inválidos" });
        }

        const tiles = await this.mapManager.getTiles(x, y, range);
        res.json(tiles);
      } catch (error) {
        log.error("mapController", `Erro ao buscar tiles: ${error.message}`);
        res.status(500).json({ error: "Erro interno do servidor" });
      }
    });

    this.app.use(router);
    log.info("mapController", "Endpoints do mapa configurados");
  }

  setupWebSocket() {
    this.io.on("connection", (socket) => {
      //WS “map-place”: Coloca um bloco no mapa
      socket.on("map-place", async (data) => {
        try {
          const { token, x, y, type } = data;

          if (
            !token ||
            typeof x !== "number" ||
            typeof y !== "number" ||
            !type
          ) {
            return;
          }

          const tokenInfo = this.tokenManager.verifyToken(token);
          if (!tokenInfo || !tokenInfo.valid) {
            return;
          }

          if (!COLORS.includes(type)) {
            socket.emit("error", { message: "Tipo de tile inválido" });
            return;
          }

          const success = await this.mapManager.placeTile(
            x,
            y,
            type,
            tokenInfo.playerName,
          );

          if (success) {
            this.io.emit("map-place", {
              x,
              y,
              type,
              playerName: tokenInfo.playerName,
            });
            log.info(
              "mapController",
              `Tile colocado em (${x}, ${y}) com cor ${type} por ${tokenInfo.playerName}`,
            );
          }
        } catch (error) {
          log.error("mapController", `Erro ao colocar tile: ${error.message}`);
        }
      });
    });

    log.info("mapController", "Endpoints WebSocket do mapa configurados");
  }
}

module.exports = { MapController };

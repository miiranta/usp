const express = require("express");
const { log } = require("../utils/colorLogging");

class StatsController {
  constructor(app, statsManager) {
    this.app = app;
    this.statsManager = statsManager;
    this.setupRoutes();
  }

  setupRoutes() {
    const router = express.Router();

    //GET /stats/:playerName: Recupera os stats de um player
    router.get("/stats/:playerName", async (req, res) => {
      try {
        const { playerName } = req.params;

        if (!playerName) {
          return res
            .status(400)
            .json({ error: "Nome do jogador é obrigatório" });
        }

        const stats = await this.statsManager.getPlayerStats(playerName);

        if (!stats) {
          return res.status(404).json({ error: "Jogador não encontrado" });
        }

        res.json(stats);
        log.info(
          "statsController",
          `Estatísticas obtidas para jogador: ${playerName}`,
        );
      } catch (error) {
        log.error(
          "statsController",
          `Erro ao obter estatísticas do jogador: ${error.message}`,
        );
        res.status(500).json({ error: "Erro interno do servidor" });
      }
    });

    this.app.use(router);
    log.info("statsController", "Endpoints de estatísticas configurados");
  }
}

module.exports = { StatsController };

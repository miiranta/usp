const { Player } = require("../domain/models/playerModel");
const { log } = require("../utils/colorLogging");

class StatsManager {
  constructor(database) {
    this.database = database;
  }

  async updateDistanceTraveled(playerName, distance) {
    try {
      const player = await Player.findByName(playerName);
      if (player) {
        await player.updateDistanceTraveled(distance);
        log.info(
          "statsManager",
          `Distância atualizada para ${playerName}: +${distance.toFixed(2)}`,
        );
      }
    } catch (error) {
      log.error(
        "statsManager",
        `Erro ao atualizar distância percorrida: ${error.message}`,
      );
    }
  }

  async updateTilesPlaced(playerName, tileType) {
    try {
      const player = await Player.findByName(playerName);
      if (player) {
        await player.updateTilesPlaced(tileType);
        log.info(
          "statsManager",
          `Tiles colocados atualizados para ${playerName}: ${tileType}`,
        );
      }
    } catch (error) {
      log.error(
        "statsManager",
        `Erro ao atualizar tiles colocados: ${error.message}`,
      );
    }
  }

  async getPlayerStats(playerName) {
    try {
      const player = await Player.findByName(playerName);

      if (!player) {
        return null;
      }

      return {
        playerName: player.playerName,
        distanceTraveled: player.stats.distanceTraveled || 0,
        tilesPlaced: Object.fromEntries(player.stats.tilesPlaced || new Map()),
      };
    } catch (error) {
      log.error(
        "statsManager",
        `Erro ao obter estatísticas do jogador: ${error.message}`,
      );
      return null;
    }
  }
}

module.exports = { StatsManager };

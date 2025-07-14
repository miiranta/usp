const { Player } = require("../domain/models/playerModel");
const { log } = require("../utils/colorLogging");

class PlayerManager {
  constructor(database, statsManager, tokenManager) {
    this.database = database;
    this.statsManager = statsManager;
    this.tokenManager = tokenManager;
  }

  async createPlayer(playerName, password) {
    try {
      const existingPlayer = await Player.findByName(playerName);
      if (existingPlayer) {
        return { success: false, message: "Jogador já existe" };
      }

      const player = new Player({ playerName });
      player.setPassword(password);
      await player.save();

      log.info("playerManager", `Jogador criado: ${playerName}`);
      return { success: true, player };
    } catch (error) {
      log.error("playerManager", `Erro ao criar jogador: ${error.message}`);
      return { success: false, message: "Falha ao criar jogador" };
    }
  }

  async authenticatePlayer(playerName, password) {
    try {
      const player = await Player.findByName(playerName);

      if (!player) {
        return { success: false, message: "Jogador não encontrado" };
      }

      if (!player.validatePassword(password)) {
        return { success: false, message: "Senha inválida" };
      }

      await player.save();

      const token = this.tokenManager.generateToken(playerName);

      log.info("playerManager", `Jogador autenticado: ${playerName}`);
      return { success: true, token: token.token, player };
    } catch (error) {
      log.error(
        "playerManager",
        `Erro ao autenticar jogador: ${error.message}`,
      );
      return { success: false, message: "Falha na autenticação" };
    }
  }

  async getPlayer(playerName) {
    try {
      const player = await Player.findByName(playerName);
      return player;
    } catch (error) {
      log.error("playerManager", `Erro ao obter jogador: ${error.message}`);
      return null;
    }
  }

  async updatePlayerPosition(playerName, x, y, previousX, previousY) {
    try {
      if (previousX !== 0 || previousY !== 0) {
        const distance = Math.sqrt(
          Math.pow(x - previousX, 2) + Math.pow(y - previousY, 2),
        );
        if (this.statsManager) {
          await this.statsManager.updateDistanceTraveled(playerName, distance);
        }
      }
      return true;
    } catch (error) {
      log.error(
        "playerManager",
        `Erro ao atualizar posição do jogador: ${error.message}`,
      );
      return false;
    }
  }
}

module.exports = { PlayerManager };

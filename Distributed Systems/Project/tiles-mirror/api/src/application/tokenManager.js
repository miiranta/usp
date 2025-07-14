const jwt = require("jsonwebtoken");
const { log } = require("../utils/colorLogging");

class TokenManager {
  constructor() {
    this.connectedPlayers = new Map(); // playerName -> { token, connectedAt, lastPing }
  }

  generateToken(playerName) {
    try {
      if (this.connectedPlayers.has(playerName)) {
        return {
          success: false,
          message: "Este nome de jogador já está em uso.",
        };
      }

      const token = jwt.sign(
        {
          playerName,
          timestamp: Date.now(),
        },
        process.env.JWT_SECRET,
        { expiresIn: "24h" },
      );

      this.connectedPlayers.set(playerName, {
        token,
        connectedAt: new Date(),
        lastPing: Date.now(),
      });
      log.success("tokenManager", `Token gerado para jogador: ${playerName}`);

      return {
        success: true,
        token,
        playerName,
      };
    } catch (error) {
      log.error("tokenManager", `Erro ao gerar token: ${error.message}`);
      return {
        success: false,
        message: "Falha ao gerar token",
      };
    }
  }

  verifyToken(token) {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const playerInfo = this.connectedPlayers.get(decoded.playerName);
      if (!playerInfo || playerInfo.token !== token) {
        return { valid: false, message: "Token não encontrado ou expirado" };
      }

      return {
        valid: true,
        playerName: decoded.playerName,
      };
    } catch (error) {
      log.error(
        "tokenManager",
        `Falha na verificação do token: ${error.message}`,
      );
      return { valid: false, message: "Token inválido" };
    }
  }

  revokeToken(playerName) {
    try {
      if (this.connectedPlayers.has(playerName)) {
        this.connectedPlayers.delete(playerName);

        log.info("tokenManager", `Token revogado para jogador: ${playerName}`);
        return playerName;
      }

      return null;
    } catch (error) {
      log.error("tokenManager", `Erro ao revogar token: ${error.message}`);
      return null;
    }
  }

  updatePlayerPing(playerName) {
    try {
      if (this.connectedPlayers.has(playerName)) {
        const playerInfo = this.connectedPlayers.get(playerName);
        playerInfo.lastPing = Date.now();
        this.connectedPlayers.set(playerName, playerInfo);
        return true;
      }
      return false;
    } catch (error) {
      log.error("tokenManager", `Erro ao atualizar ping do jogador: ${error.message}`);
      return false;
    }
  }

  getDisconnectedPlayers(timeoutMs = 60000) {
    const disconnectedPlayers = [];
    const now = Date.now();
    
    for (const [playerName, playerInfo] of this.connectedPlayers.entries()) {
      if (now - playerInfo.lastPing > timeoutMs) {
        disconnectedPlayers.push(playerName);
      }
    }
    
    return disconnectedPlayers;
  }

  removeDisconnectedPlayers(timeoutMs = 60000) {
    const disconnectedPlayers = this.getDisconnectedPlayers(timeoutMs);
    
    disconnectedPlayers.forEach(playerName => {
      this.connectedPlayers.delete(playerName);
      log.info("tokenManager", `Jogador removido por timeout: ${playerName}`);
    });
    
    return disconnectedPlayers;
  }

  isNameAvailable(playerName) {
    return !this.connectedPlayers.has(playerName);
  }
}

module.exports = { TokenManager };

const express = require("express");
const { log } = require("../utils/colorLogging");

class PlayerController {
  constructor(app, io, playerManager, tokenManager) {
    this.app = app;
    this.io = io;
    this.playerManager = playerManager;
    this.tokenManager = tokenManager;
    this.setupRoutes();
    this.setupWebSocket();
    this.startDisconnectionChecker();
  }

  setupRoutes() {
    const router = express.Router();

    //GET /player/:playerName: Verifica se um player já existe
    router.get("/player/:playerName", async (req, res) => {
      try {
        const { playerName } = req.params;
        if (!playerName || playerName.trim().length === 0) {
          return res
            .status(400)
            .json({ error: "Nome do jogador é obrigatório" });
        }

        const trimmedName = playerName.trim();
        if (trimmedName.length < 2 || trimmedName.length > 20) {
          return res.status(400).json({
            error: "Nome do jogador deve ter entre 2 e 20 caracteres",
            taken: true,
          });
        }
        const isCurrentlyConnected =
          !this.tokenManager.isNameAvailable(trimmedName);

        const existingPlayer = await this.playerManager.getPlayer(trimmedName);
        const existsInDatabase = !!existingPlayer;
        res.status(200).json({
          taken: isCurrentlyConnected,
          available: !isCurrentlyConnected && !existsInDatabase,
          existsInDatabase: existsInDatabase,
          currentlyConnected: isCurrentlyConnected,
        });
      } catch (error) {
        log.error(
          "playerController",
          `Erro ao verificar disponibilidade do nome: ${error.message}`,
        );
        res
          .status(500)
          .json({ error: "Erro interno do servidor", taken: true });
      }
    });

    //POST /player: Cria um novo player, se disponível
    router.post("/player", async (req, res) => {
      try {
        const { playerName, password } = req.body;

        if (!playerName) {
          return res
            .status(400)
            .json({ error: "Nome de jogador é obrigatório." });
        }

        if (playerName.trim().length < 2 || playerName.trim().length > 20) {
          return res.status(400).json({
            error: "Nomes de jogador devem ter entre 2 e 20 caracteres.",
          });
        }

        const trimmedName = playerName.trim();

        if (!this.tokenManager.isNameAvailable(trimmedName)) {
          return res
            .status(409)
            .json({ error: "Este nome de jogador já está conectado." });
        }
        if (password) {
          if (password.length < 4) {
            return res
              .status(400)
              .json({ error: "Senha deve ter pelo menos 4 caracteres." });
          }

          const result = await this.playerManager.createPlayer(
            trimmedName,
            password,
          );
          if (!result.success) {
            return res.status(409).json({ error: result.message });
          }

          log.success(
            "playerController",
            `Jogador criado no banco de dados: ${trimmedName}`,
          );
        }

        const tokenResult = this.tokenManager.generateToken(trimmedName);

        if (tokenResult.success) {
          res.status(201).json({
            success: true,
            token: tokenResult.token,
            playerName: tokenResult.playerName,
          });
          log.success(
            "playerController",
            `Sessão do jogador criada: ${tokenResult.playerName}`,
          );
        } else {
          res.status(409).json({ error: tokenResult.message });
        }
      } catch (error) {
        log.error(
          "playerController",
          `Erro ao criar jogador: ${error.message}`,
        );
        res.status(500).json({ error: "Erro interno do servidor" });
      }
    });

    //POST /player/authenticate: Autentica um jogador existente
    router.post("/player/authenticate", async (req, res) => {
      try {
        const { playerName, password } = req.body;

        if (!playerName || !password) {
          return res
            .status(400)
            .json({ error: "Nome de jogador e senha são obrigatórios." });
        }

        const trimmedName = playerName.trim();

        if (!this.tokenManager.isNameAvailable(trimmedName)) {
          return res
            .status(409)
            .json({ error: "Este jogador já está conectado." });
        }

        const result = await this.playerManager.authenticatePlayer(
          trimmedName,
          password,
        );
        if (!result.success) {
          const statusCode = result.message === "Player not found" ? 404 : 401;
          return res.status(statusCode).json({ error: result.message });
        }

        res.status(200).json({
          success: true,
          token: result.token,
          playerName: result.player.playerName,
        });
        log.success(
          "playerController",
          `Jogador autenticado: ${result.player.playerName}`,
        );
      } catch (error) {
        log.error(
          "playerController",
          `Erro ao autenticar jogador: ${error.message}`,
        );
        res.status(500).json({ error: "Erro interno do servidor" });
      }
    });
    this.app.use(router);
    log.info("playerController", "Endpoints do jogador configurados");
  }

  startDisconnectionChecker() {
    setInterval(() => {
      const disconnectedPlayers = this.tokenManager.removeDisconnectedPlayers(60000);
      
      disconnectedPlayers.forEach(playerName => {
        //WS “player-remove”: Remove o player (timeout)
        this.io.emit("player-remove", {
          playerName: playerName,
        });
        log.info(
          "playerController",
          `Jogador removido por timeout de ping: ${playerName}`,
        );
      });
    }, 30000);
  }

  setupWebSocket() {
    this.io.on("connection", (socket) => {
      log.info("playerController", `Cliente conectado: ${socket.id}`);

      let socketPlayerName = null;
      let lastPlayerPosition = { x: 0, y: 0 };

      //WS “player-update”: Atualiza a posição de um player
      socket.on("player-update", async (data) => {
        try {
          const { token, x, y, hasMoved } = data;

          if (!token || typeof x !== "number" || typeof y !== "number") {
            return;
          }

          const tokenInfo = this.tokenManager.verifyToken(token);
          if (!tokenInfo || !tokenInfo.valid) {
            return;
          }

          socketPlayerName = tokenInfo.playerName;

          this.tokenManager.updatePlayerPing(tokenInfo.playerName);

          this.playerManager
            .updatePlayerPosition(
              tokenInfo.playerName,
              x,
              y,
              lastPlayerPosition.x,
              lastPlayerPosition.y,
            )
            .catch((err) => {
              log.error(
                "playerController",
                `Erro ao atualizar posição do jogador: ${err.message}`,
              );
            });

          lastPlayerPosition = { x, y };

          socket.broadcast.emit("player-update", {
            playerName: tokenInfo.playerName,
            x,
            y,
            hasMoved: hasMoved || false,
          });
        } catch (error) {
          log.error(
            "playerController",
            `Erro ao lidar com atualização do jogador: ${error.message}`,
          );
        }
      });

      //WS “player-remove”: Remove o player (Disconnect)
      socket.on("disconnect", () => {
        try {
          if (socketPlayerName) {
            const removedPlayerName =
              this.tokenManager.revokeToken(socketPlayerName);
            if (removedPlayerName) {
              socket.broadcast.emit("player-remove", {
                playerName: removedPlayerName,
              });
              log.info(
                "playerController",
                `Jogador desconectado: ${removedPlayerName} (${socket.id})`,
              );
            }
          } else {
            log.info("playerController", `Cliente desconectado: ${socket.id}`);
          }
        } catch (error) {
          log.error(
            "playerController",
            `Erro ao lidar com desconexão: ${error.message}`,
          );
        }
      });
    });

    log.info("playerController", "Endpoints WebSocket configurados");
  }
}

module.exports = { PlayerController };

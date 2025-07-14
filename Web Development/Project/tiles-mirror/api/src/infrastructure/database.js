const mongoose = require("mongoose");
const { log } = require("../utils/colorLogging");

class Database {
  constructor() {
    this.connection = null;
  }

  async connect() {
    try {
      const connectionString = process.env.MONGODB_CONNECTION_STRING;
      if (!connectionString) {
        throw new Error(
          "MONGODB_CONNECTION_STRING não está definida nas variáveis de ambiente",
        );
      }
      this.connection = await mongoose.connect(connectionString);

      log.success("database", "Conectado ao MongoDB com sucesso");

      mongoose.connection.on("error", (err) => {
        log.error("database", `Erro de conexão com MongoDB: ${err.message}`);
      });

      mongoose.connection.on("disconnected", () => {
        log.info("database", "MongoDB desconectado");
      });
      return this.connection;
    } catch (error) {
      log.error("database", `Falha ao conectar ao MongoDB: ${error.message}`);
      throw error;
    }
  }

  async disconnect() {
    try {
      if (this.connection) {
        await mongoose.connection.close();
        log.info("database", "Conexão com banco de dados fechada com sucesso");
      }
    } catch (error) {
      log.error(
        "database",
        `Erro ao fechar conexão com banco de dados: ${error.message}`,
      );
      throw error;
    }
  }

  getConnection() {
    return this.connection;
  }
}

module.exports = { Database };

const { StatsController } = require("../view/statsController");
const { MapController } = require("../view/mapController");
const { PlayerController } = require("../view/playerController");

const { MapManager } = require("./mapManager");
const { PlayerManager } = require("./playerManager");
const { StatsManager } = require("./statsManager");
const { TokenManager } = require("./tokenManager");

class AppManager {
  constructor(app, io, database) {
    this.tokenManager = new TokenManager();
    this.statsManager = new StatsManager(database);
    this.mapManager = new MapManager(database, this.statsManager);
    this.playerManager = new PlayerManager(
      database,
      this.statsManager,
      this.tokenManager,
    );

    this.statsController = new StatsController(app, this.statsManager);
    this.mapController = new MapController(
      app,
      io,
      this.mapManager,
      this.tokenManager,
    );
    this.playerController = new PlayerController(
      app,
      io,
      this.playerManager,
      this.tokenManager,
    );
  }
}

module.exports = { AppManager };

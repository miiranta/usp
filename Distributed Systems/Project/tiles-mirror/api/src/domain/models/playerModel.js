const mongoose = require("mongoose");
const crypto = require("crypto");

const playerSchema = new mongoose.Schema(
  {
    playerName: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      minlength: 2,
      maxlength: 20,
    },
    passwordHash: {
      type: String,
      required: true,
    },
    stats: {
      distanceTraveled: {
        type: Number,
        default: 0,
      },
      tilesPlaced: {
        type: Map,
        of: Number,
        default: {},
      },
    },
  },
  {
    timestamps: true,
  },
);

playerSchema.methods.setPassword = function (password) {
  this.passwordHash = crypto
    .createHash("sha256")
    .update(password)
    .digest("hex");
};

playerSchema.methods.validatePassword = function (password) {
  const hash = crypto.createHash("sha256").update(password).digest("hex");
  return this.passwordHash === hash;
};

playerSchema.statics.findByName = function (playerName) {
  return this.findOne({ playerName: playerName.trim() });
};

playerSchema.methods.updateDistanceTraveled = function (distance) {
  this.stats.distanceTraveled = (this.stats.distanceTraveled || 0) + distance;
  return this.save();
};

playerSchema.methods.updateTilesPlaced = function (tileType) {
  if (!this.stats.tilesPlaced) {
    this.stats.tilesPlaced = new Map();
  }
  const currentCount = this.stats.tilesPlaced.get(tileType) || 0;
  this.stats.tilesPlaced.set(tileType, currentCount + 1);
  return this.save();
};

const Player = mongoose.model("Player", playerSchema);

module.exports = {
  Player,
};

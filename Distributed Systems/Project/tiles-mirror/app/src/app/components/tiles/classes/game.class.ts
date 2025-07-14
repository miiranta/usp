import { TICKS_PER_SECOND } from '../constants/game-config.consts';
import { DrawManager } from './managers/draw-manager.class';
import { KeyManager } from './managers/key-manager.class';
import { MultiplayerManager } from './managers/multiplayer-manager.class';
import { Player } from './player/player.class';
import { TileMap } from './map/tile-map.class';

export class Game {
  drawManager: DrawManager = new DrawManager();
  keyManager: KeyManager = new KeyManager();
  multiplayerManager: MultiplayerManager = new MultiplayerManager();

  map!: TileMap;
  player!: Player;
  canvas: any;
  apiMap: any;
  apiPlayer: any;
  playerService: any;
  playerName: string;

  constructor(
    canvas: any,
    apiMap: any,
    apiPlayer: any,
    playerService: any,
    playerName: string,
  ) {
    this.playerName = playerName;
    if (this.playerName == '' || this.playerName == undefined) {
      playerName = Math.random().toString(36).substring(2, 15);
      this.playerName = playerName;
    }

    this.canvas = canvas.nativeElement;
    if (!this.canvas) {
      return;
    }

    this.apiMap = apiMap;
    if (!this.apiMap) {
      return;
    }

    this.apiPlayer = apiPlayer;
    if (!this.apiPlayer) {
      return;
    }

    this.playerService = playerService;
    if (!this.apiMap || !this.apiPlayer) {
      return;
    }

    this.map = new TileMap(this.apiMap, this.apiPlayer, this.playerService);
    if (!this.map) {
      return;
    }

    this.player = new Player(this.playerName);
    if (!this.player) {
      return;
    }

    this.drawManager.setGameTarget(this);
    this.multiplayerManager.setGameTarget(this);

    setInterval(this.gameLoop.bind(this), 1000 / TICKS_PER_SECOND); // Game logic
    setInterval(this.apiLoop.bind(this), 50);
    this.drawLoop();
  }

  gameLoop() {
    this.player.updateSpeed(this.keyManager.keyMap);
    this.player.updatePosition();

    this.multiplayerManager.updatePlayerPositions();
  }

  drawLoop() {
    this.drawManager.clear();
    this.drawManager.draw();

    requestAnimationFrame(() => this.drawLoop());
  }
  apiLoop() {
    this.multiplayerManager.sendPlayerUpdate();
  }
}

import { Game } from '../game.class';
import { Tile } from '../map/tile.class';
import { MultiplayerPlayer } from '../../interfaces/multiplayer-player.interface';
import {
  MIN_UPDATE_INTERVAL,
  MIN_MOVEMENT_THRESHOLD,
  PLAYER_MOVE_DURATION,
} from '../../constants/game-config.consts';

export class MultiplayerManager {
  private game!: Game;
  private lastSentPosition: { x: number; y: number } = { x: 0, y: 0 };
  private lastSentTime: number = 0;

  public all_players: Map<string, MultiplayerPlayer> = new Map();
  constructor() {}

  setGameTarget(game: Game) {
    this.game = game;

    this.listenPlayerUpdates();
    this.listenPlayerRemove();
    this.listenMapPlace();

    setInterval(() => {
      this.sendPlayerUpdate(true);
    }, 1000 * 10);
  }

  sendPlayerUpdate(bypassMovementCheck: boolean = false) {
    const player_coords = this.game.player.getPositionFloat();
    const currentTime = Date.now();

    if (currentTime - this.lastSentTime < MIN_UPDATE_INTERVAL) {
      return;
    }

    const dx = Math.abs(player_coords.x - this.lastSentPosition.x);
    const dy = Math.abs(player_coords.y - this.lastSentPosition.y);

    if ((dx < MIN_MOVEMENT_THRESHOLD && dy < MIN_MOVEMENT_THRESHOLD) && !bypassMovementCheck) {
      return;
    }
    const hasMoved = (dx >= MIN_MOVEMENT_THRESHOLD || dy >= MIN_MOVEMENT_THRESHOLD);

    const token = this.game.playerService.getJwtToken();
    
    if (token) {
      this.game.apiPlayer.sendPlayerUpdate(
        token,
        player_coords.x,
        player_coords.y,
        hasMoved
      );
      this.lastSentPosition = { x: player_coords.x, y: player_coords.y };
      this.lastSentTime = currentTime;
    }
  }

  updatePlayerPositions() {
    const currentTime = Date.now();
    this.all_players.forEach((player) => {
      const elapsed = currentTime - player.moveStartTime;
      const progress = Math.min(elapsed / player.moveDuration, 1.0);

      const easedProgress = 1 - Math.pow(1 - progress, 3);

      player.x =
        player.startX + (player.targetX - player.startX) * easedProgress;
      player.y =
        player.startY + (player.targetY - player.startY) * easedProgress;

      if (progress >= 1.0) {
        player.x = player.targetX;
        player.y = player.targetY;
      }
    });
  }

  listenPlayerUpdates() {
    this.game.apiPlayer.onPlayerUpdate().subscribe((data: any) => {
      if (data.playerName === this.game.player.playerName) {
        return;
      }
      const targetX = Math.round(data.x);
      const targetY = Math.round(data.y);

      if (this.all_players.has(data.playerName)) {
        const player = this.all_players.get(data.playerName)!;

        if (player.targetX !== targetX || player.targetY !== targetY) {
          player.startX = player.x;
          player.startY = player.y;
          player.targetX = targetX;
          player.targetY = targetY;
          player.moveStartTime = Date.now();
          player.moveDuration = PLAYER_MOVE_DURATION;
        }

        player.last_update = Date.now();
        
        if (data.hasMoved) {
          player.last_movement = Date.now();
        }
      } else {
        const newPlayer: MultiplayerPlayer = {
          playerName: data.playerName,
          x: targetX,
          y: targetY,
          startX: targetX,
          startY: targetY,
          targetX: targetX,
          targetY: targetY,
          moveStartTime: Date.now(),
          moveDuration: PLAYER_MOVE_DURATION,
          last_update: Date.now(),
          last_movement: data.hasMoved ? Date.now() : 0,
          randomRgbColor: `rgb(${Math.floor(Math.random() * 256)}, ${Math.floor(
            Math.random() * 256,
          )}, ${Math.floor(Math.random() * 256)})`,
        };
        this.all_players.set(data.playerName, newPlayer);
      }
    });
  }

  listenPlayerRemove() {
    this.game.apiPlayer.onPlayerRemove().subscribe((data: any) => {
      if (data.playerName && this.all_players.has(data.playerName)) {
        this.all_players.delete(data.playerName);
        console.log(`Jogador ${data.playerName} desconectado`);
      }
    });
  }

  listenMapPlace() {
    this.game.apiMap.onMapPlace().subscribe((data: any) => {
      const tile = new Tile(data.x, data.y, data.type);
      this.game.map.placeTileLocal(tile.x, tile.y, tile.type);
    });
  }
}

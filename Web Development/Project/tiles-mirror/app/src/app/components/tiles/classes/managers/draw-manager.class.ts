import {
  TILE_SIZE,
  DRAW_ALL_MAP,
  RENDER_DISTANCE,
} from '../../constants/game-config.consts';
import { Game } from '../game.class';
import { Tile } from '../map/tile.class';
import { MultiplayerPlayer } from '../../interfaces/multiplayer-player.interface';

export class DrawManager {
  private ctx: any;

  private scaling_factor: number = 1;
  private scale_speed: number = 0;

  private game!: Game;

  private last_draw_time: number = 0;
  fps: number = 0;
  constructor() {}

  setGameTarget(game: Game) {
    this.game = game;
    this.ctx = this.game.canvas.getContext('2d');
  }
  setScalingFactor(scrollIndex: number) {
    const maxScalingFactor = 2;
    const minScalingFactor = 0.2;

    this.scale_speed *= 0.9;
    if (Math.abs(this.scale_speed) < 0.001) {
      this.scale_speed = 0;
    }

    this.scale_speed += scrollIndex / 4000;

    this.scaling_factor += this.scale_speed;

    if (this.scaling_factor > maxScalingFactor) {
      this.scaling_factor = maxScalingFactor;
    }
    if (this.scaling_factor < minScalingFactor) {
      this.scaling_factor = minScalingFactor;
    }
  }

  calculateFPS() {
    const now = performance.now();
    const elapsed = now - this.last_draw_time;
    this.last_draw_time = now;

    return 1000 / elapsed;
  }

  clear() {
    if (!this.game) {
      return;
    }

    const ctx = this.game.canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.game.canvas.width, this.game.canvas.height);
    }
  }
  draw() {
    if (!this.game) {
      return;
    }

    this.fps = this.calculateFPS();

    this.setScalingFactor(this.game.keyManager.scrollIndex);
    this.game.keyManager.scrollIndex = 0;

    if (this.ctx) {
      this.ctx.scale(this.scaling_factor, this.scaling_factor);
    }
    const translateX =
      this.game.canvas.width / 2 / this.scaling_factor - TILE_SIZE / 2;
    const translateY =
      this.game.canvas.height / 2 / this.scaling_factor - TILE_SIZE / 2;

    if (this.ctx) {
      this.ctx.translate(translateX, translateY);
    }

    this.drawMap();

    this.drawOtherPlayers();

    this.drawPlayer();
  }
  drawPlayer() {
    if (this.ctx) {
      this.ctx.strokeStyle = 'black';
      this.ctx.lineWidth = 4;
      this.ctx.strokeRect(0, 0, TILE_SIZE, TILE_SIZE);
    }
  }

  drawOtherPlayers() {
    if (this.ctx) {
      this.game.multiplayerManager.all_players.forEach(
        (player: MultiplayerPlayer) => {
          if (player.playerName == this.game.player.playerName) return;

          const player_coords = { x: player.x, y: player.y };
          const relative_x =
            player_coords.x - this.game.player.getPositionFloat().x;
          const relative_y =
            player_coords.y - this.game.player.getPositionFloat().y;

          this.ctx.strokeStyle = player.randomRgbColor;
          this.ctx.lineWidth = 4;
          this.ctx.strokeRect(
            relative_x * TILE_SIZE,
            relative_y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
          );

          this.ctx.fillStyle = 'black';
          this.ctx.textAlign = 'center';
          this.ctx.font = '20px Arial';
          this.ctx.fillText(
            player.playerName,
            relative_x * TILE_SIZE + TILE_SIZE / 2,
            relative_y * TILE_SIZE - TILE_SIZE / 8,
          );
        },
      );
    }
  }

  drawMap() {
    if (this.ctx) {
      const player_coords = this.game.player.getPosition();

      if (DRAW_ALL_MAP) {
        var render_distance_horizontal =
          Math.floor(
            (this.game.canvas.width / TILE_SIZE / 2) *
              (1 / this.scaling_factor),
          ) + 1;
        var render_distance_vertical =
          Math.floor(
            (this.game.canvas.height / TILE_SIZE / 2) *
              (1 / this.scaling_factor),
          ) + 1;
      } else {
        var render_distance_horizontal = RENDER_DISTANCE;
        var render_distance_vertical = RENDER_DISTANCE;
      }

      for (
        let x = player_coords.x - render_distance_horizontal;
        x < player_coords.x + render_distance_horizontal + 1;
        x++
      ) {
        for (
          let y = player_coords.y - render_distance_vertical;
          y < player_coords.y + render_distance_vertical + 1;
          y++
        ) {
          const tile = this.game.map.getTile(x, y);

          const player_coords_float = this.game.player.getPositionFloat();
          const x_relative = x - player_coords_float.x;
          const y_relative = y - player_coords_float.y;

          this.drawTile(tile, x_relative, y_relative);
        }
      }
    }
  }
  drawTile(tile: Tile, relative_x: number, relative_y: number) {
    if (this.ctx) {
      this.ctx.fillStyle = tile.type;

      this.ctx.fillRect(
        relative_x * TILE_SIZE,
        relative_y * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
      );

      if (tile.x == 0 && tile.y == 0) {
        this.drawSpawnPoint(relative_x, relative_y);
      }
    }
  }

  drawSpawnPoint(relative_x: number, relative_y: number) {
    this.ctx.fillStyle = 'red';
    this.ctx.fillRect(
      relative_x * TILE_SIZE + TILE_SIZE / 4,
      relative_y * TILE_SIZE + TILE_SIZE / 4,
      TILE_SIZE / 2,
      TILE_SIZE / 2,
    );
  }
}

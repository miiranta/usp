import { Component, inject, Input, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ApiFetchService } from '../../services/api-fetch.service';

const TICKS_PER_SECOND: number = 1024;
const TILE_SIZE: number = 100;
const CHUNK_SIZE: number = 10;

const RENDER_DISTANCE: number = 20;

const DRAW_ALL_MAP: boolean = true;

const COLORS: string[] = [
  'white',
  'lightgray',
  'gray',
  'darkgray',
  'black',
  'red',
  'green',
  'blue',
  'yellow',
  'purple',
  'orange'
];

@Component({
  selector: 'tiles',
  imports: [CommonModule],
  templateUrl: './tiles.component.html',
  styleUrl: './tiles.component.scss'
})
export class TilesComponent {
  @ViewChild('tiles') tiles: any;
  @Input() playerId: string = "";

  api: ApiFetchService = inject(ApiFetchService);

  coords: any = { x: 0, y: 0 };
  fps: number = 0;
  colors: string[] = COLORS;
  selected_color: string = COLORS[0];
  game?: Game = undefined;

  ngAfterViewInit() {
    this.game = new Game(this.tiles, this.api, this.playerId);

    this.canvasSetSize();
    window.addEventListener('resize', this.canvasSetSize.bind(this));

    setInterval(() => {
        this.coords = this.game?.player.getPosition();
        this.fps = Math.round(this.game?.drawManager.fps ?? 0);
      }, 1000 / 10);
  }

  canvasSetSize() {
    if (this.tiles) {
      this.tiles.nativeElement.width = window.innerWidth;
      this.tiles.nativeElement.height = window.innerHeight;
    }
  }

  selectColor(color: string) {
    this.selected_color = color;
  }

  placeTile() {
    const x = this.coords.x;
    const y = this.coords.y;

    if (this.selected_color) {
      this.game?.map.placeTile(x, y, this.selected_color);
    }
  }

}

class Game {
  drawManager: DrawManager = new DrawManager();
  keyManager: KeyManager = new KeyManager();
  multiplayerManager: MultiplayerManager = new MultiplayerManager();
  
  map!: TileMap;
  player!: Player;
  canvas: any;
  api: any;
  playerId: string;

  constructor(canvas: any, api: any, playerId: string) {
    this.playerId = playerId;
    if(this.playerId == '' || this.playerId == undefined) {
      playerId = Math.random().toString(36).substring(2, 15);
      this.playerId = playerId;
    }
    
    this.canvas = canvas.nativeElement;
    if(!this.canvas) { return; }

    this.api = api;
    if(!this.api) { return; }

    this.map = new TileMap(this.api);
    if(!this.map) { return; }

    this.player = new Player(this.playerId);
    if(!this.player) { return; }

    this.drawManager.setGameTarget(this);
    this.multiplayerManager.setGameTarget(this);

    setInterval(this.gameLoop.bind(this), 1000 / TICKS_PER_SECOND);
    setInterval(this.drawLoop.bind(this), 0);
    setInterval(this.apiLoop.bind(this), 5);
  }

  gameLoop() {
    // Update player speed
    this.player.updateSpeed(this.keyManager.keyMap);
    this.player.updatePosition();
  }

  drawLoop() {
    this.drawManager.clear();
    this.drawManager.draw();
  }

  apiLoop() {
    this.multiplayerManager.sendPlayerPosition();
    this.multiplayerManager.filterOldPlayers();
    this.multiplayerManager.listenTilePlaced();
  }
}

class MultiplayerManager {
  private game!: Game;

  public all_players: Map<string, Player> = new Map();

  constructor() { 
    // Do nothing here, please
  }

  setGameTarget(game: Game) {
    this.game = game;

    this.listenPlayerPositions();
  }

  // Player position
  sendPlayerPosition() {
    const player_coords = this.game.player.getPositionFloat();
    this.game.api.sendPlayerPosition(this.game.player.playerId, player_coords.x, player_coords.y);
  }

  listenPlayerPositions() {
    this.game.api.on('playerPosition').subscribe((data: any) => {
      // Test if the player is already in the map
      if (this.all_players.has(data.playerId)) {
        var player = this.all_players.get(data.playerId)!;
        player.x = data.x;
        player.y = data.y;
        player.last_update = Date.now();
      } else {
        const newPlayer = new Player(data.playerId);
        newPlayer.x = data.x;
        newPlayer.y = data.y;
        newPlayer.last_update = Date.now();
        this.all_players.set(data.playerId, newPlayer);
      }
    });
  }

  filterOldPlayers() {
    const now = Date.now();
    const timeout = 1000; // 1 seconds

    this.all_players.forEach((player, playerId) => {
      if (now - player.last_update > timeout) {
        this.all_players.delete(playerId);
      }
    });
  }

  // Tile update
  listenTilePlaced() {
    this.game.api.on('tilePlaced').subscribe((data: any) => {
      const tile = new Tile(data.x, data.y, data.type);
      this.game.map.setTile(tile.x, tile.y, tile.type);
    });
  }

}

class KeyManager {
  keyMap: Map<string, boolean> = new Map();
  scrollIndex: number = 0;

  constructor() {
    // Listen for keydown events
    window.addEventListener('keydown', this.keydown.bind(this));
    // Listen for keyup events
    window.addEventListener('keyup', this.keyup.bind(this));
    // Listen for scroll events
    window.addEventListener('wheel', this.scroll.bind(this));
  }
  
  keydown(event: KeyboardEvent) {
    this.keyMap.set(event.key, true);
  }

  keyup(event: KeyboardEvent) {
    this.keyMap.set(event.key, false);
  }

  scroll(event: Event) {
    const wheelEvent = event as WheelEvent;
    this.scrollIndex = this.scrollIndex + wheelEvent.deltaY;
  }

}

class DrawManager {
  private ctx: any;

  private scaling_factor: number = 1;
  private scale_speed: number = 0;

  private game!: Game;

  private last_draw_time: number = 0;
  fps: number = 0;

  constructor() {
    // Do nothing here, please
  }

  setGameTarget(game: Game) {
    this.game = game
    this.ctx = this.game.canvas.getContext('2d');
  }

  setScalingFactor(scrollIndex: number) {
    const maxScalingFactor = 2;
    const minScalingFactor = 0.06;

    //Speed down
    this.scale_speed *= 0.9;
    if(Math.abs(this.scale_speed) < 0.001) {
      this.scale_speed = 0;
    }

    //Speed up
    this.scale_speed += scrollIndex / 4000;

    //Apply scaling factor
    this.scaling_factor += this.scale_speed;

    //Limit scaling factor
    if(this.scaling_factor > maxScalingFactor) {
      this.scaling_factor = maxScalingFactor;
    }
    if(this.scaling_factor < minScalingFactor) {
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
    if(!this.game) { return; }

    const ctx = this.game.canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, this.game.canvas.width, this.game.canvas.height);
    }
  }

  draw() {
    if(!this.game) { return; }

    // Calculate FPS
    this.fps = this.calculateFPS();

    // Set scaling factor
    this.setScalingFactor(this.game.keyManager.scrollIndex);
    this.game.keyManager.scrollIndex = 0;

    // Apply scaling of CanvasRenderingContext2D
    if(this.ctx){this.ctx.scale(this.scaling_factor, this.scaling_factor);}

    //Center the player on the screen - corrected for scaling
    const translateX = (this.game.canvas.width / 2) / this.scaling_factor - TILE_SIZE / 2;
    const translateY = (this.game.canvas.height / 2) / this.scaling_factor - TILE_SIZE / 2;

    // Apply translation of CanvasRenderingContext2D
    if(this.ctx){this.ctx.translate(translateX, translateY);}
    
    // Draw the map
    this.drawMap();

    // Draw other players
    this.drawOtherPlayers();

    // Draw the player
    this.drawPlayer();
  }

  drawPlayer() {
    if (this.ctx) {
      
      // A square border in the middle of the screen
      this.ctx.strokeStyle = 'black';
      this.ctx.lineWidth = 4;
      this.ctx.strokeRect(0, 0, TILE_SIZE, TILE_SIZE);

    }

  }

  drawOtherPlayers() {
    if (this.ctx) {
      this.game.multiplayerManager.all_players.forEach((player) => {
        if(player.playerId == this.game.player.playerId) return;

        const player_coords = player.getPositionFloat();
        const relative_x = player_coords.x - this.game.player.getPositionFloat().x;
        const relative_y = player_coords.y - this.game.player.getPositionFloat().y;

        this.ctx.strokeStyle = player.randomRgbColor;
        this.ctx.lineWidth = 4;
        this.ctx.strokeRect(relative_x * TILE_SIZE, relative_y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

        // Draw player ID
        this.ctx.fillStyle = "black";
        this.ctx.textAlign = 'center';
        this.ctx.font = '20px Arial';
        this.ctx.fillText(player.playerId, relative_x * TILE_SIZE + TILE_SIZE / 2, relative_y * TILE_SIZE - TILE_SIZE/8);
      });
    }
  }

  drawMap() {
    if (this.ctx) {
      
      // Render distance - A square with the player in the middle
      const player_coords = this.game.player.getPosition();

      // Draw only the tiles in the render distance
      for (let x = player_coords.x - RENDER_DISTANCE; x < player_coords.x + RENDER_DISTANCE + 1; x++) {
        for (let y = player_coords.y - RENDER_DISTANCE; y < player_coords.y + RENDER_DISTANCE + 1; y++) {
          const tile = this.game.map.getTile(x, y);

          if(DRAW_ALL_MAP) continue;
          // Coords relative to the player - center is 0, 0
          const player_coords_float = this.game.player.getPositionFloat();
          const x_relative = x - player_coords_float.x;
          const y_relative = y - player_coords_float.y;

          this.drawTile(tile, x_relative, y_relative);
        }
      }

      if(DRAW_ALL_MAP) {
        // Draw all tiles
        this.game.map.map.forEach(tile => {
          const player_coords_float = this.game.player.getPositionFloat();
          const x_relative = tile.x - player_coords_float.x;
          const y_relative = tile.y - player_coords_float.y;

          this.drawTile(tile, x_relative, y_relative);
        });
      }
      
    }
  }

  drawTile(tile: Tile, relative_x: number, relative_y: number) {
    if (this.ctx) {

      const x = tile.x;
      const y = tile.y;
      
      this.ctx.fillStyle = tile.type;
      
      this.ctx.fillRect(relative_x * TILE_SIZE, relative_y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

      if(tile.x == 0 && tile.y == 0) {
        this.drawSpawnPoint(relative_x, relative_y);
      }

    }
  }

  drawSpawnPoint(relative_x: number, relative_y: number) {
    this.ctx.fillStyle = 'red';
    this.ctx.fillRect(relative_x * TILE_SIZE + TILE_SIZE/4, relative_y * TILE_SIZE + TILE_SIZE/4, (TILE_SIZE/2), (TILE_SIZE/2));
  }

}

class Player {
  private x_speed: number = 0;
  private y_speed: number = 0;
  public x: number = 0;
  public y: number = 0;
  public playerId: string;
  public last_update: number = 0;
  public randomRgbColor: string = '';

  constructor(playerId: string) {
    this.playerId = playerId;
    this.randomRgbColor = `rgb(${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)})`;
  }

  getPosition() {
    const x_int = Math.round(this.x);
    const y_int = Math.round(this.y);

    return { x: x_int, y: y_int };
  }

  getPositionFloat() {
    return { x: this.x, y: this.y };
  }

  getSpeed() {
    const x_speed_int = Math.round(this.x_speed);
    const y_speed_int = Math.round(this.y_speed);

    return { x: x_speed_int, y: y_speed_int };
  }

  updateSpeed(keyMap: Map<string, boolean>) {
    // Speed down
    this.x_speed *= 0.85;
    this.y_speed *= 0.85;
    if(Math.abs(this.x_speed) < 0.1) {
      this.x_speed = 0;
    }
    if(Math.abs(this.y_speed) < 0.1) {
      this.y_speed = 0;
    }

    // Speed up
    if (keyMap.get('w') || keyMap.get('ArrowUp')) {
      this.y_speed -= 2;
    }
    if (keyMap.get('a') || keyMap.get('ArrowLeft')) {
      this.x_speed -= 2;
    }
    if (keyMap.get('s') || keyMap.get('ArrowDown')) {
      this.y_speed += 2;
    }
    if (keyMap.get('d') || keyMap.get('ArrowRight')) {
      this.x_speed += 2;
    }
  }

  updatePosition() {
    this.x += this.x_speed / TICKS_PER_SECOND;
    this.y += this.y_speed / TICKS_PER_SECOND;

    // Ints
    let x_int = Math.round(this.x);
    let y_int = Math.round(this.y);

    // Sum 1 to speed direction
    if (this.x_speed > 0) {
      x_int += 1;
    }else if (this.x_speed < 0) {
      x_int -= 1;
    }
    if (this.y_speed > 0) {
      y_int += 1;
    }else if (this.y_speed < 0) {
      y_int -= 1;
    }

    // Update float to walk a little bit in int direction
    this.x = x_int * 0.02 + this.x * 0.98;
    this.y = y_int * 0.02 + this.y * 0.98;
  }

}

class TileMap {
  private api: any;
  constructor(api: any) {this.api = api;}

  chunkFetchPending: string[] = [];

  map: Map<string, Tile> = new Map();

  getTile(x: number, y: number) {
    return this.fetchTile(x, y);
  }

  fetchTile(x: number, y: number) {
    const key = `${x},${y}`;
    if (this.map.has(key)) {
      return this.map.get(key)!;
    }

    // Create a new tile and add it to the map
    const center_x = Math.floor(x / CHUNK_SIZE) * CHUNK_SIZE;
    const center_y = Math.floor(y / CHUNK_SIZE) * CHUNK_SIZE;
    const chunk_key = `${center_x},${center_y}`;

    if (!this.chunkFetchPending.includes(chunk_key)) {
      this.chunkFetchPending.push(chunk_key);

      this.api.getMapTiles(center_x, center_y, CHUNK_SIZE).then(async (response: Response) => {
        if(response.status == 200) {
          const json = response.json();
          json.then((data) => {
            for (const tileData of data) {
              const tile = new Tile(tileData.x, tileData.y, tileData.type);
              this.map.set(`${tile.x},${tile.y}`, tile);
            }
            this.chunkFetchPending = this.chunkFetchPending.filter((key) => key !== chunk_key);
          });
        }
      });
    }

    return new Tile(x, y);
  }

  placeTile(x: number, y: number, type: string) {
    const key = `${x},${y}`;
    if (this.map.has(key)) {
      const tile = this.map.get(key)!;
      tile.type = type;
    } else {
      // Create a new tile and add it to the map
      const tile = new Tile(x, y);
      tile.type = type;
      this.map.set(key, tile);
    }

    this.api.emit('tilePlaced', { x, y, type });
    this.api.putMapTile(x, y, type);
  }

  setTile(x: number, y: number, type: string) {
    const key = `${x},${y}`;
    if (this.map.has(key)) {
      const tile = this.map.get(key)!;
      tile.type = type;
    } else {
      // Create a new tile and add it to the map
      const tile = new Tile(x, y);
      tile.type = type;
      this.map.set(key, tile);
    }
  }

}

class Tile {
  type = 'white';
  x = 0;
  y = 0;
  
  constructor(x:number, y:number, type:string='notset') {
    this.x = x;
    this.y = y;

    if (type == 'notset') {
        // Set tile type based on coordinates - grid of white and light gray
        if ((x % 2 + 2) % 2 === (y % 2 + 2) % 2) {
        this.type = 'white';
        } else {
        this.type = 'lightgray';
        }
    } else {
        this.type = type;
    }

  }

}

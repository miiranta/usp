import { CHUNK_SIZE } from '../../constants/game-config.consts';
import { Tile } from './tile.class';

export class TileMap {
  private apiMap: any;
  private apiPlayer: any;
  private playerService: any;
  private pendingPlacements: Set<string> = new Set();

  constructor(apiMap: any, apiPlayer: any, playerService: any) {
    this.apiMap = apiMap;
    this.apiPlayer = apiPlayer;
    this.playerService = playerService;
  }

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

    const center_x = Math.floor(x / CHUNK_SIZE) * CHUNK_SIZE;
    const center_y = Math.floor(y / CHUNK_SIZE) * CHUNK_SIZE;
    const chunk_key = `${center_x},${center_y}`;

    if (!this.chunkFetchPending.includes(chunk_key)) {
      this.chunkFetchPending.push(chunk_key);

      this.apiMap
        .getMapTiles(center_x, center_y, CHUNK_SIZE)
        .then(async (response: Response) => {
          if (response.status == 200) {
            const json = response.json();
            json.then((data) => {
              for (const tileData of data) {
                const tile = new Tile(tileData.x, tileData.y, tileData.type);
                this.map.set(`${tile.x},${tile.y}`, tile);
              }
              this.chunkFetchPending = this.chunkFetchPending.filter(
                (key) => key !== chunk_key,
              );
            });
          }
        });
    }

    return new Tile(x, y);
  }

  placeTile(x: number, y: number, type: string) {
    const key = `${x},${y}`;

    if (!this.map.has(key)) return;

    const currentTile = this.map.get(key);
    if (currentTile && currentTile.type === type) return;
    if (this.pendingPlacements.has(key)) return;

    const token = this.playerService.getJwtToken();
    if (!token) return;

    this.pendingPlacements.add(key);
    
    setTimeout(() => {
      this.pendingPlacements.delete(key);
    }, 500);

    this.apiMap.sendMapPlace(token, x, y, type);
  }

  placeTileLocal(x: number, y: number, type: string) {
    const key = `${x},${y}`;

    if (!this.map.has(key)) return;

    if (this.map.get(key)?.type === type) return;

    this.map.set(key, new Tile(x, y, type));
    
    this.pendingPlacements.delete(`${x},${y}`);
  }
}

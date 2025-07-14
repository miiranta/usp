export interface PlayerStats {
  playerName: string;
  distanceTraveled: number;
  tilesPlaced: { [tileType: string]: number };
}

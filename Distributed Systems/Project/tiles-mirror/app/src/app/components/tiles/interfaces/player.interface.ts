export interface Player {
  x: number;
  y: number;
  playerName: string;
  last_update: number;
  last_movement: number;
  randomRgbColor: string;
}

export interface Position {
  x: number;
  y: number;
}

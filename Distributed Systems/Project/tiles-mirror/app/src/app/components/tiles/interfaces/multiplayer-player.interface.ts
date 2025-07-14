import { Player } from './player.interface';

export interface MultiplayerPlayer extends Player {
  targetX: number;
  targetY: number;
  startX: number;
  startY: number;
  moveStartTime: number;
  moveDuration: number;
}

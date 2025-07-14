import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-game-stats',
  imports: [],
  templateUrl: './game-stats.component.html',
  styleUrl: './game-stats.component.scss',
})
export class GameStatsComponent {
  @Input() coords: any = { x: 0, y: 0 };
  @Input() fps: number = 0;
  @Input() playerName: string = '';
}

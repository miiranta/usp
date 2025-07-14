import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { COLORS } from './enums/colors.model';
import { GameStatsComponent } from './components/game-stats/game-stats.component';
import { GameCanvasComponent } from './components/game-canvas/game-canvas.component';
import { ColorSelectorComponent } from './components/color-selector/color-selector.component';
import { ListPlayersComponent } from './components/list-players/list-players.component';

@Component({
  selector: 'tiles',
  imports: [
    CommonModule,
    GameStatsComponent,
    GameCanvasComponent,
    ColorSelectorComponent,
    ListPlayersComponent,
  ],
  templateUrl: './tiles.component.html',
  styleUrl: './tiles.component.scss',
})
export class TilesComponent {
  @Input() playerName: string = '';

  coords: any = { x: 0, y: 0 };
  fps: number = 0;
  colors: string[] = COLORS;
  selected_color: string = COLORS[0];
  game: any = null;

  onCoordsChanged(coords: any) {
    this.coords = coords;
  }

  onFpsChanged(fps: number) {
    this.fps = fps;
  }

  onGameInitialized(game: any) {
    this.game = game;
  }

  onColorSelected(color: string) {
    this.selected_color = color;
  }
}

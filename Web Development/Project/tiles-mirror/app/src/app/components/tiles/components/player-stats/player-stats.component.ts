import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnInit,
  OnChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiStatsService } from '../../../../services/api-stats.service';
import { PlayerStats } from '../../interfaces/player-stats.interface';

@Component({
  selector: 'app-player-stats',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './player-stats.component.html',
  styleUrl: './player-stats.component.scss',
})
export class PlayerStatsComponent implements OnInit, OnChanges {
  @Input() playerName!: string;
  @Input() isVisible: boolean = false;
  @Output() closeStats = new EventEmitter<void>();

  playerStats: PlayerStats | null = null;
  isLoading: boolean = false;
  error: string | null = null;

  constructor(private apiStatsService: ApiStatsService) {}

  ngOnInit() {
    if (this.isVisible && this.playerName) {
      this.loadPlayerStats();
    }
  }

  ngOnChanges() {
    if (this.isVisible && this.playerName) {
      this.loadPlayerStats();
    }

    if (!this.isVisible) {
      this.playerStats = null;
      this.error = null;
    }
  }

  loadPlayerStats() {
    this.isLoading = true;
    this.error = null;
    this.playerStats = null; // Clear previous stats

    this.apiStatsService
      .getPlayerStats(this.playerName)
      .then((stats) => {
        setTimeout(() => {
          this.playerStats = stats;
          this.isLoading = false;
        }, 300);
      })
      .catch((error) => {
        this.error = 'Falha ao carregar estatísticas do jogador';
        this.isLoading = false;
        console.error('Erro ao carregar estatísticas do jogador:', error);
      });
  }

  onClose() {
    this.closeStats.emit();
  }
  getTileTypes(): string[] {
    if (!this.playerStats?.tilesPlaced) return [];
    return Object.keys(this.playerStats.tilesPlaced).sort(
      (a, b) =>
        this.playerStats!.tilesPlaced[b] - this.playerStats!.tilesPlaced[a],
    );
  }
  getTotalTilesPlaced(): number {
    if (!this.playerStats?.tilesPlaced) return 0;
    return Object.values(this.playerStats.tilesPlaced).reduce(
      (sum: number, count: number) => sum + count,
      0,
    );
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }

  formatDistance(distance: number): string {
    return distance.toFixed(2);
  }
}

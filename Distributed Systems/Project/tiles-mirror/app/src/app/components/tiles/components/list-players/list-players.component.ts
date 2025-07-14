import {
  Component,
  Input,
  OnInit,
  OnDestroy,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { MultiplayerManager } from '../../classes/managers/multiplayer-manager.class';
import { MultiplayerPlayer } from '../../interfaces/multiplayer-player.interface';
import { PlayerStatsComponent } from '../player-stats/player-stats.component';

@Component({
  selector: 'app-list-players',
  imports: [CommonModule, PlayerStatsComponent],
  templateUrl: './list-players.component.html',
  styleUrl: './list-players.component.scss',
})
export class ListPlayersComponent implements OnInit, OnDestroy {
  @Input() multiplayerManager!: MultiplayerManager;
  @Input() currentPlayerName: string = '';

  private updateInterval: any;

  showStatsModal: boolean = false;
  selectedPlayerName: string = '';

  constructor(private cdr: ChangeDetectorRef) {}
  ngOnInit() {
    this.updateInterval = setInterval(() => {
      this.cdr.detectChanges();
    }, 1000);
  }

  ngOnDestroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
  get allPlayers(): MultiplayerPlayer[] {
    if (!this.multiplayerManager) {
      return [];
    }
    return Array.from(this.multiplayerManager.all_players.values());
  }

  get totalPlayers(): number {
    return this.allPlayers.length + 1;
  }

  isPlayerActive(player: MultiplayerPlayer): boolean {
    const thirtySecondsAgo = Date.now() - 30000;
    
    const hasRecentPing = player.last_update > thirtySecondsAgo;
    
    if (!hasRecentPing) {
      return false;
    }
    
    if (player.last_movement === 0) {
      return false;
    }
    
    const hasRecentMovement = player.last_movement > thirtySecondsAgo;
    return hasRecentMovement;
  }

  getPlayerStatus(player: MultiplayerPlayer): string {
    return this.isPlayerActive(player) ? 'online' : 'inactive';
  }

  openPlayerStats(playerName: string) {
    if (this.showStatsModal) {
      this.showStatsModal = false;
      setTimeout(() => {
        this.selectedPlayerName = playerName;
        this.showStatsModal = true;
      }, 100);
    } else {
      this.selectedPlayerName = playerName;
      this.showStatsModal = true;
    }
  }

  closePlayerStats() {
    this.showStatsModal = false;
    this.selectedPlayerName = '';
  }
}

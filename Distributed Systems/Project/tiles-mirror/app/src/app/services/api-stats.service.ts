import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { PlayerStats } from '../components/tiles/interfaces/player-stats.interface';

const BASE_URL = `${environment.BASE_URL}:${environment.PORT}`;

@Injectable({
  providedIn: 'root',
})
export class ApiStatsService {
  constructor() {}

  async getPlayerStats(playerName: string): Promise<PlayerStats> {
    const response = await fetch(`${BASE_URL}/stats/${playerName}`);
    if (!response.ok) {
      throw new Error(
        `Erro ao buscar estatísticas do jogador: ${response.statusText}`,
      );
    }
    const data: PlayerStats = await response.json();
    return data;
  }
}

import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { WebsocketService } from './websocket.service';

const BASE_URL = `${environment.BASE_URL}:${environment.PORT}`;

@Injectable({
  providedIn: 'root',
})
export class ApiPlayerService {
  constructor(private websocketService: WebsocketService) {}

  async checkPlayerNameAvailability(playerName: string): Promise<any> {
    return await fetch(`${BASE_URL}/player/${encodeURIComponent(playerName)}`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });
  }

  async createPlayerWithPassword(
    playerName: string,
    password: string,
  ): Promise<any> {
    return await fetch(`${BASE_URL}/player`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        playerName,
        password,
      }),
    });
  }

  async authenticatePlayer(playerName: string, password: string): Promise<any> {
    return await fetch(`${BASE_URL}/player/authenticate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        playerName,
        password,
      }),
    });
  }

  sendPlayerUpdate(token: string, x: number, y: number, hasMoved?: boolean) {
    this.websocketService.emit('player-update', { token, x, y, hasMoved });
  }

  onPlayerUpdate(): Observable<any> {
    return this.websocketService.on('player-update');
  }

  onPlayerRemove(): Observable<any> {
    return this.websocketService.on('player-remove');
  }
}

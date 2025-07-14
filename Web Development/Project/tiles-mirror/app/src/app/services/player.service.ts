import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class PlayerService {
  private playerName = '';
  private jwtToken = '';

  constructor() {}

  setPlayerName(name: string): void {
    this.playerName = name;
  }

  getPlayerName(): string {
    return this.playerName || '';
  }

  clearPlayerName(): void {
    this.playerName = '';
    this.jwtToken = '';
  }

  setJwtToken(token: string): void {
    this.jwtToken = token;
  }

  getJwtToken(): string {
    return this.jwtToken || '';
  }
}

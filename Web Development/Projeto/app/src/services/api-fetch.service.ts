import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { Observable } from 'rxjs';

const BASE_URL = 'http://localhost:3000';

@Injectable({
  providedIn: 'root'
})
export class ApiFetchService {

  constructor() {
    this.socket = io(BASE_URL);
  }

  // WebSocket
  private socket: Socket;

  emit(event: string, data: any) {
    this.socket.emit(event, data);
  }

  on(event: string): Observable<any> {
    return new Observable((observer) => {
      this.socket.on(event, (data) => {
        observer.next(data);
      });

      // Handle cleanup
      return () => {
        this.socket.off(event);
      };
    });
  }

  sendPlayerPosition(playerId: string, x: number, y: number) {
    this.socket.emit('playerPosition', { playerId, x, y });
  }

  // REST
  async getMapTiles(x: number, y: number, render: number){
    return await fetch(`${BASE_URL}/map/${x}/${y}/${render}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    })
  }

  async putMapTile(x: number, y: number, type: string){
    return await fetch(`${BASE_URL}/map/${x}/${y}/${type}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
    })
  }

 

}

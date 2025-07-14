import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { WebsocketService } from './websocket.service';

const BASE_URL = `${environment.BASE_URL}:${environment.PORT}`;

@Injectable({
  providedIn: 'root',
})
export class ApiMapService {
  constructor(private websocketService: WebsocketService) {}

  async getMapTiles(x: number, y: number, range: number) {
    return await fetch(`${BASE_URL}/map/${x}/${y}/${range}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
    });
  }

  sendMapPlace(token: string, x: number, y: number, type: string) {
    this.websocketService.emit('map-place', { token, x, y, type });
  }

  onMapPlace(): Observable<any> {
    return this.websocketService.on('map-place');
  }
}

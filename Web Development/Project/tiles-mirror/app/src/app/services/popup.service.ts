import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export interface PopupData {
  title: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  duration?: number; // in milliseconds, 0 means no auto-close
}

@Injectable({
  providedIn: 'root',
})
export class PopupService {
  private popupSubject = new BehaviorSubject<PopupData | null>(null);
  public popup$ = this.popupSubject.asObservable();

  constructor() {}

  show(data: PopupData): void {
    this.popupSubject.next(data);

    if (data.duration && data.duration > 0) {
      setTimeout(() => {
        this.hide();
      }, data.duration);
    }
  }

  success(title: string, message: string, duration: number = 3000): void {
    this.show({ title, message, type: 'success', duration });
  }

  error(title: string, message: string, duration: number = 5000): void {
    this.show({ title, message, type: 'error', duration });
  }

  warning(title: string, message: string, duration: number = 4000): void {
    this.show({ title, message, type: 'warning', duration });
  }

  info(title: string, message: string, duration: number = 3000): void {
    this.show({ title, message, type: 'info', duration });
  }

  hide(): void {
    this.popupSubject.next(null);
  }
}

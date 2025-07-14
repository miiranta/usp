import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class LoadingService {
  private loadingSubject = new BehaviorSubject<boolean>(false);
  private messageSubject = new BehaviorSubject<string>('Carregando...');

  public loading$ = this.loadingSubject.asObservable();
  public message$ = this.messageSubject.asObservable();

  constructor() {}

  show(message: string = 'Carregando...'): void {
    this.messageSubject.next(message);
    this.loadingSubject.next(true);
  }

  hide(): void {
    this.loadingSubject.next(false);
  }

  isLoading(): boolean {
    return this.loadingSubject.value;
  }

  setMessage(message: string): void {
    this.messageSubject.next(message);
  }
}

import { Injectable } from '@angular/core';
import { PopupComponent } from '../components/popup/popup.component';
import { Observable, Subscriber } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class PopupService {
  private popupComponents: PopupComponent[] = [];
  private currentObserver?: Subscriber<string>;

  constructor() {
    (window as any).res = (res: string) => {
      this.res(res);
    };
  }

  subscribePopupComponent(component: PopupComponent) {
    if (!this.popupComponents.includes(component)) {
      this.popupComponents.push(component);
    }
  }

  open(data: string, timer?: number, type?: string): Observable<string> {
    this.popupComponents.forEach((component) => {
      component.showPopup(data, timer, type);
    });

    return new Observable<string>((observer) => {
      this.currentObserver = observer;
    });
  }

  close(): Observable<string> {
    this.popupComponents.forEach((component) => {
      component.hidePopup();
    });

    return new Observable<string>((observer) => {
      this.currentObserver = observer;
    });
  }

  res(result: string) {
    if (this.currentObserver) {
      this.currentObserver.next(result);
      this.currentObserver.complete();
      this.currentObserver = undefined;
    }
  }
}

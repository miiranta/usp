import { Injectable } from '@angular/core';
import { LoadingComponent } from '../components/loading/loading.component';

@Injectable({
  providedIn: 'root',
})
export class LoadingService {
  loadingComponents: LoadingComponent[] = [];

  constructor() {}

  subscribeLoadingComponent(component: LoadingComponent) {
    if (!this.loadingComponents.includes(component))
      this.loadingComponents.push(component);
  }

  open(text?: string) {
    this.loadingComponents.forEach((component) => {
      if (component) {
        component.isLoading = true;
        component.loadingText = text || '';
      }
    });
  }

  close() {
    this.loadingComponents.forEach((component) => {
      if (component) {
        component.isLoading = false;
        component.loadingText = '';
      }
    });
  }

  updateText(text: string) {
    this.loadingComponents.forEach((component) => {
      if (component) {
        component.loadingText = text;
      }
    });
  }
}

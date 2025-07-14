import { Component, inject, Input } from '@angular/core';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-loading',
  imports: [],
  templateUrl: './loading.component.html',
  styleUrl: './loading.component.scss',
})
export class LoadingComponent {
  @Input() isLoading: boolean = false;
  loadingText: string = '';

  private loadingService: LoadingService = inject(LoadingService);

  constructor() {
    this.loadingService.subscribeLoadingComponent(this);
  }
}

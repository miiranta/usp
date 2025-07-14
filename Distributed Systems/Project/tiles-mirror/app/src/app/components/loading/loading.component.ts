import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-loading',
  imports: [CommonModule],
  templateUrl: './loading.component.html',
  styleUrl: './loading.component.scss',
})
export class LoadingComponent implements OnInit, OnDestroy {
  show: boolean = false;
  message: string = 'Carregando...';

  private loadingSubscription: Subscription = new Subscription();
  private messageSubscription: Subscription = new Subscription();

  constructor(private loadingService: LoadingService) {}

  ngOnInit(): void {
    this.loadingSubscription = this.loadingService.loading$.subscribe(
      (loading) => (this.show = loading),
    );

    this.messageSubscription = this.loadingService.message$.subscribe(
      (message) => (this.message = message),
    );
  }

  ngOnDestroy(): void {
    this.loadingSubscription.unsubscribe();
    this.messageSubscription.unsubscribe();
  }
}

import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { PopupService, PopupData } from '../../services/popup.service';

@Component({
  selector: 'app-popup',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './popup.component.html',
  styleUrl: './popup.component.scss',
})
export class PopupComponent implements OnInit, OnDestroy {
  currentPopup: PopupData | null = null;
  private subscription: Subscription = new Subscription();

  constructor(private popupService: PopupService) {}

  ngOnInit(): void {
    this.subscription = this.popupService.popup$.subscribe((popup) => {
      this.currentPopup = popup;
    });
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  close(): void {
    this.popupService.hide();
  }

  getIconClass(): string {
    if (!this.currentPopup) return '';

    switch (this.currentPopup.type) {
      case 'success':
        return 'icon-success';
      case 'error':
        return 'icon-error';
      case 'warning':
        return 'icon-warning';
      case 'info':
        return 'icon-info';
      default:
        return '';
    }
  }
}

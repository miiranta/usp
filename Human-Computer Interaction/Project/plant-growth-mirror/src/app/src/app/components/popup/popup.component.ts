import { Component, inject, AfterViewInit, HostListener } from '@angular/core';
import { PopupService } from '../../services/popup.service';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-popup',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './popup.component.html',
  styleUrls: ['./popup.component.scss'],
})
export class PopupComponent {
  private popupService = inject(PopupService);
  private sanitizer = inject(DomSanitizer);

  popupData: SafeHtml | null = null;
  popupVisible = false;
  popupType: string | null = null;
  isHiding = false;
  timer = 0;

  constructor() {
    this.popupService.subscribePopupComponent(this);
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape' && this.popupVisible) {
      this.hidePopup();
      this.popupService.res('dismissed');
    }
  }

  showPopup(data: string, timerMs?: number, type?: string) {
    const dataSkipSanitization = this.sanitizer.bypassSecurityTrustHtml(data);

    this.popupData = dataSkipSanitization;
    this.popupType = type || null;
    this.popupVisible = true;
    this.isHiding = false;

    if (timerMs && timerMs > 0) {
      this.timer = timerMs;
    }
  }

  onBarFinished() {
    this.hidePopup(true);
  }

  hidePopup(animated = false) {
    if (animated) {
      this.isHiding = true;
      setTimeout(() => this.finalHide(), 300);
    } else {
      this.finalHide();
    }
  }

  private finalHide() {
    this.popupVisible = false;
    this.isHiding = false;
    this.popupData = null;
    this.timer = 0;
  }

  getPopupStyle() {
    switch (this.popupType) {
      case 'error':
        return {
          'background-color': 'rgba(255, 82, 82, 0.8)',
          color: '#d8000c',
        };
      case 'success':
        return {
          'background-color': 'rgba(204, 255, 204, 0.8)',
          color: '#4caf50',
        };
      case 'info':
        return {
          'background-color': 'rgba(224, 247, 250, 0.8)',
          color: '#00796b',
        };
      default:
        return {};
    }
  }
}

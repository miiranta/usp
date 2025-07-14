import { CommonModule } from '@angular/common';
import { Component, Input, OnInit, OnDestroy, OnChanges } from '@angular/core';

@Component({
  selector: 'app-spinner',
  imports: [CommonModule],
  templateUrl: './spinner.component.html',
  styleUrl: './spinner.component.scss',
})
export class SpinnerComponent implements OnInit, OnDestroy, OnChanges {
  @Input() showSpinner: boolean = false;
  @Input() showText: boolean = false;
  @Input() spinnerText: string = '';
  @Input() showProgress: boolean = false;
  @Input() showProgressText: boolean = true;

  progress: number = 0;
  private intervalId: any;

  ngOnInit() {
    if (this.showProgress && this.showSpinner) {
      this.startProgressSimulation();
    }
  }

  ngOnDestroy() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }

  ngOnChanges() {
    if (this.showProgress && this.showSpinner && !this.intervalId) {
      this.startProgressSimulation();
    } else if (!this.showSpinner && this.intervalId) {
      this.stopProgressSimulation();
    }
  }

  private startProgressSimulation() {
    this.progress = 0;
    this.intervalId = setInterval(() => {
      const remaining = 100 - this.progress;
      const increment = Math.random() * (remaining * 0.1);

      if (increment < 0.5 && this.progress > 85) {
        this.progress += Math.random() * 0.3;
      } else {
        this.progress += increment;
      }

      if (this.progress > 99) {
        this.progress = 99;
      }
    }, 100);
  }

  private stopProgressSimulation() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  completeProgress() {
    this.progress = 100;
    this.stopProgressSimulation();
  }

  roundProgress(): number {
    return Math.round(this.progress);
  }
}

import {
  Component,
  Input,
  OnInit,
  AfterViewInit,
  ElementRef,
  ViewChild,
  OnChanges,
  SimpleChanges,
  OnDestroy,
  inject,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, registerables, TimeScale } from 'chart.js';
import 'chartjs-adapter-date-fns';
import { Photo } from '../../models/photo';
import { Collection } from '../../models/collection';
import { ApiService } from '../../services/api.service';
import { SpinnerComponent } from '../spinner/spinner.component';

@Component({
  selector: 'app-show-graphs',
  standalone: true,
  imports: [CommonModule, SpinnerComponent],
  templateUrl: './show-graphs.component.html',
  styleUrl: './show-graphs.component.scss',
})
export class ShowGraphsComponent
  implements OnInit, AfterViewInit, OnChanges, OnDestroy
{
  @ViewChild('heightChart', { static: false })
  heightChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('widthChart', { static: false })
  widthChartRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('areaChart', { static: false })
  areaChartRef?: ElementRef<HTMLCanvasElement>;

  private heightChart?: Chart;
  private widthChart?: Chart;
  private areaChart?: Chart;

  private apiService = inject(ApiService);
  @Input() photos: Photo[] = [];
  @Input() collection: Collection | null = null;

  processedPhotos: Photo[] = [];
  hasData = false;
  hasHeightData = false;
  hasWidthData = false;
  hasAreaData = false;
  heightCount = 0;
  widthCount = 0;
  areaCount = 0;
  mightLoad = true;

  heightDateMode: 'real' | 'equal' = 'equal';
  widthDateMode: 'real' | 'equal' = 'equal';
  areaDateMode: 'real' | 'equal' = 'equal';

  constructor() {
    Chart.register(...registerables);
  }
  ngOnInit(): void {
    this.setup();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['collection'] || changes['photos']) {
      this.mightLoad = true;

      if (this.photos && this.photos.length > 0) {
        this.updateData();
        setTimeout(() => this.recreateCharts(), 100);
      } else if (changes['collection'] && this.collection) {
        this.setup();
      } else {
        this.mightLoad = false;
      }
    }
  }

  ngAfterViewInit(): void {
    setTimeout(() => {
      if (
        this.hasData &&
        !this.heightChart &&
        !this.widthChart &&
        !this.areaChart
      ) {
        this.createCharts();
      }
    }, 200);
  }

  ngOnDestroy(): void {
    this.destroyCharts();
  }
  private updateData(): void {
    this.processedPhotos = this.photos
      .filter(
        (photo) =>
          photo.result &&
          (photo.result.height !== undefined ||
            photo.result.width !== undefined ||
            photo.result.area !== undefined),
      )
      .sort((a, b) => {
        if (!a.createdAt) return 1;
        if (!b.createdAt) return -1;
        return (
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
      });

    this.hasData = this.processedPhotos.length > 0;
    this.hasHeightData = this.processedPhotos.some(
      (p) => p.result?.height !== undefined,
    );
    this.hasWidthData = this.processedPhotos.some(
      (p) => p.result?.width !== undefined,
    );
    this.hasAreaData = this.processedPhotos.some(
      (p) => p.result?.area !== undefined,
    );

    this.heightCount = this.processedPhotos.filter(
      (p) => p.result?.height !== undefined,
    ).length;
    this.widthCount = this.processedPhotos.filter(
      (p) => p.result?.width !== undefined,
    ).length;
    this.areaCount = this.processedPhotos.filter(
      (p) => p.result?.area !== undefined,
    ).length;

    this.mightLoad = false;
  }

  private recreateCharts(): void {
    this.destroyCharts();
    setTimeout(() => this.createCharts(), 100);
  }
  private createCharts(): void {
    if (!this.hasData) return;

    this.destroyCharts();

    if (this.hasHeightData && this.heightChartRef?.nativeElement) {
      this.createHeightChart();
    }

    if (this.hasWidthData && this.widthChartRef?.nativeElement) {
      this.createWidthChart();
    }

    if (this.hasAreaData && this.areaChartRef?.nativeElement) {
      this.createAreaChart();
    }
  }
  toggleHeightDateMode(): void {
    if (this.heightChart) {
      this.heightChart.destroy();
      this.heightChart = undefined;
      setTimeout(() => this.createHeightChart(), 50);
    }
  }

  toggleWidthDateMode(): void {
    if (this.widthChart) {
      this.widthChart.destroy();
      this.widthChart = undefined;
      setTimeout(() => this.createWidthChart(), 50);
    }
  }

  toggleAreaDateMode(): void {
    if (this.areaChart) {
      this.areaChart.destroy();
      this.areaChart = undefined;
      setTimeout(() => this.createAreaChart(), 50);
    }
  }
  setHeightDateMode(mode: 'real' | 'equal'): void {
    if (this.heightDateMode !== mode) {
      this.heightDateMode = mode;
      this.toggleHeightDateMode();
    }
  }

  setWidthDateMode(mode: 'real' | 'equal'): void {
    if (this.widthDateMode !== mode) {
      this.widthDateMode = mode;
      this.toggleWidthDateMode();
    }
  }

  setAreaDateMode(mode: 'real' | 'equal'): void {
    if (this.areaDateMode !== mode) {
      this.areaDateMode = mode;
      this.toggleAreaDateMode();
    }
  }
  private getDatesForMode(mode: 'real' | 'equal'): string[] {
    if (mode === 'equal') {
      return this.processedPhotos.map((photo, index) =>
        photo.createdAt
          ? new Date(photo.createdAt).toLocaleDateString()
          : `Photo ${index + 1}`,
      );
    } else {
      return this.processedPhotos.map((photo) =>
        photo.createdAt
          ? new Date(photo.createdAt).toISOString()
          : new Date().toISOString(),
      );
    }
  }
  private getDataForMode(
    mode: 'real' | 'equal',
    values: (number | null)[],
  ): any[] {
    if (mode === 'real') {
      return this.processedPhotos.map((photo, index) => ({
        x: photo.createdAt ? new Date(photo.createdAt) : new Date(),
        y: values[index],
      }));
    } else {
      return values as any[];
    }
  }
  private createHeightChart(): void {
    if (!this.heightChartRef?.nativeElement || this.heightChart) return;

    const heightValues = this.processedPhotos.map(
      (photo) => photo.result?.height || null,
    );
    const isRealMode = this.heightDateMode === 'real';
    const dates = this.getDatesForMode(this.heightDateMode);
    const data = this.getDataForMode(this.heightDateMode, heightValues);

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: isRealMode ? undefined : dates,
        datasets: [
          {
            label: 'Height (pixels)',
            data: data,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            spanGaps: true,
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: false,
          },
          legend: {
            display: false,
          },
        },
        scales: {
          x: isRealMode
            ? {
                type: 'time',
                time: {
                  unit: 'day',
                },
                title: {
                  display: true,
                  text: 'Date',
                },
              }
            : {
                title: {
                  display: true,
                  text: 'Photo Sequence',
                },
              },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Height (pixels)',
            },
          },
        },
      },
    };

    this.heightChart = new Chart(this.heightChartRef.nativeElement, config);
  }

  private createWidthChart(): void {
    if (!this.widthChartRef?.nativeElement || this.widthChart) return;

    const widthValues = this.processedPhotos.map(
      (photo) => photo.result?.width || null,
    );
    const isRealMode = this.widthDateMode === 'real';
    const dates = this.getDatesForMode(this.widthDateMode);
    const data = this.getDataForMode(this.widthDateMode, widthValues);

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: isRealMode ? undefined : dates,
        datasets: [
          {
            label: 'Width (pixels)',
            data: data,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            spanGaps: true,
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: false,
          },
          legend: {
            display: false,
          },
        },
        scales: {
          x: isRealMode
            ? {
                type: 'time',
                time: {
                  unit: 'day',
                },
                title: {
                  display: true,
                  text: 'Date',
                },
              }
            : {
                title: {
                  display: true,
                  text: 'Photo Sequence',
                },
              },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Width (pixels)',
            },
          },
        },
      },
    };

    this.widthChart = new Chart(this.widthChartRef.nativeElement, config);
  }

  private createAreaChart(): void {
    if (!this.areaChartRef?.nativeElement || this.areaChart) return;

    const areaValues = this.processedPhotos.map(
      (photo) => photo.result?.area || null,
    );
    const isRealMode = this.areaDateMode === 'real';
    const dates = this.getDatesForMode(this.areaDateMode);
    const data = this.getDataForMode(this.areaDateMode, areaValues);

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: isRealMode ? undefined : dates,
        datasets: [
          {
            label: 'Area (sq pixels)',
            data: data,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            spanGaps: true,
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: false,
          },
          legend: {
            display: false,
          },
        },
        scales: {
          x: isRealMode
            ? {
                type: 'time',
                time: {
                  unit: 'day',
                },
                title: {
                  display: true,
                  text: 'Date',
                },
              }
            : {
                title: {
                  display: true,
                  text: 'Photo Sequence',
                },
              },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Area (square pixels)',
            },
          },
        },
      },
    };

    this.areaChart = new Chart(this.areaChartRef.nativeElement, config);
  }

  private destroyCharts(): void {
    if (this.heightChart) {
      this.heightChart.destroy();
      this.heightChart = undefined;
    }
    if (this.widthChart) {
      this.widthChart.destroy();
      this.widthChart = undefined;
    }
    if (this.areaChart) {
      this.areaChart.destroy();
      this.areaChart = undefined;
    }
  }
  private async setup() {
    if (this.photos && this.photos.length > 0) {
      this.updateData();
      setTimeout(() => this.createCharts(), 100);
      return;
    }

    if (!this.collection) {
      await this.getCollection();
    }
    if (this.collection) {
      await this.getPhotos();
    } else {
      this.mightLoad = false;
    }

    if (this.photos.length > 0 && this.hasData) {
      setTimeout(() => this.createCharts(), 100);
    }
  }

  private async getCollection() {
    const collectionId = window.location.pathname.split('/').pop();
    if (!collectionId || collectionId.trim() === '') {
      return;
    }
    const allCollections = await this.apiService.getCollections();
    this.collection = allCollections.find((c) => c.id === collectionId) || null;
    if (!this.collection) {
      return;
    }
  }
  private async getPhotos() {
    if (
      !this.collection ||
      !this.collection.photoIds ||
      this.collection.photoIds.length === 0
    ) {
      this.photos = [];
      this.updateData();
      return;
    }

    try {
      const loaded: Photo[] = [];
      for (const id of this.collection.photoIds) {
        const photo = await this.apiService.getPhoto(id);
        if (photo) {
          loaded.push(photo);
        }
      }
      this.photos = loaded;
      this.updateData();
    } catch (error: any) {
      this.mightLoad = false;
    }
  }
}

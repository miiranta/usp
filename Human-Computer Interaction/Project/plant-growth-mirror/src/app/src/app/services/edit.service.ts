import { Injectable } from '@angular/core';
import { Photo } from '../models/photo';

@Injectable({
  providedIn: 'root',
})
export class EditService {
  private stagedPhotos: Photo[] = [];
  private collectionId: string | null = null;
  private granularity = 20;
  private threshold = 0.05;
  private previousRoute: string | null = null;

  constructor() {}

  getStagedPhotos(): Photo[] {
    return this.stagedPhotos;
  }

  pushStagedPhoto(photo: Photo): void {
    this.stagedPhotos.push(photo);
  }

  clearStagedPhotos(): void {
    this.stagedPhotos = [];
  }

  setGranularity(value: number): void {
    this.granularity = value;
  }

  getGranularity(): number {
    return this.granularity;
  }

  setThreshold(value: number): void {
    this.threshold = value;
  }

  getThreshold(): number {
    return this.threshold;
  }

  setCollectionId(id: string | null): void {
    this.collectionId = id;
  }

  getCollectionId(): string | null {
    return this.collectionId;
  }

  setPreviousRoute(route: string | null): void {
    this.previousRoute = route;
  }

  getPreviousRoute(): string | null {
    return this.previousRoute;
  }
}

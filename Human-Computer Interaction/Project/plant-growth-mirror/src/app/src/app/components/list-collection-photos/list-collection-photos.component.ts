import {
  Component,
  inject,
  Input,
  OnInit,
  Output,
  SimpleChanges,
  OnChanges,
  EventEmitter,
} from '@angular/core';
import { Photo } from '../../models/photo';
import { ViewPhotoComponent } from '../view-photo/view-photo.component';
import { PopupService } from '../../services/popup.service';
import { ApiService } from '../../services/api.service';
import { Router } from '@angular/router';
import { Collection } from '../../models/collection';
import { LoadingService } from '../../services/loading.service';
import { EditService } from '../../services/edit.service';
import { SpinnerComponent } from '../spinner/spinner.component';

@Component({
  selector: 'app-list-collection-photos',
  standalone: true,
  imports: [ViewPhotoComponent, SpinnerComponent],
  templateUrl: './list-collection-photos.component.html',
  styleUrls: ['./list-collection-photos.component.scss'],
})
export class ListCollectionPhotosComponent implements OnInit, OnChanges {
  @Input() collection: Collection | null = null;
  @Output() photosLoaded = new EventEmitter<Photo[]>();
  @Output() collectionUpdated = new EventEmitter<Collection>();

  photos: Photo[] = [];
  currentPhoto: Photo | null = null;
  mightLoad: boolean = true;
  private apiService = inject(ApiService);
  private popupService = inject(PopupService);
  private editService = inject(EditService);
  private router = inject(Router);

  ngOnInit() {
    this.setup();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['collection']) {
      this.setup();
    }
  }

  selectPhoto(photo: Photo) {
    if (!photo) return;
    this.currentPhoto = photo;
  }

  goToCollectionEditPage() {
    if (!this.collection) return;
    this.editService.clearStagedPhotos();
    this.photos.forEach((photo) => this.editService.pushStagedPhoto(photo));
    this.editService.setPreviousRoute(this.router.url);
    this.router.navigate(['/editPhoto']);
  }

  async refreshListing() {
    if (!this.collection) return;

    try {
      const collectionId = this.collection.id;
      const allCollections = await this.apiService.getCollections();
      this.collection =
        allCollections.find((c) => c.id === collectionId) || null;

      if (!this.collection) {
        this.router.navigate(['/']);
        return;
      }

      this.photos = this.photos.filter(
        (photo) => this.collection?.photoIds?.includes(photo.id) || false,
      );

      if (
        !this.currentPhoto ||
        !this.photos.find((p) => p.id === this.currentPhoto?.id)
      ) {
        this.currentPhoto = this.photos.length > 0 ? this.photos[0] : null;
      }

      this.photosLoaded.emit(this.photos);

      this.collectionUpdated.emit(this.collection);
    } catch (error) {
      this.popupService.open(
        'Failed to refresh collection: ' +
          (error instanceof Error ? error.message : 'Unknown error'),
        5000,
        'error',
      );
    }
  }

  onRefreshListing() {
    this.refreshListing().catch((error) => {});
  }

  private async setup() {
    this.mightLoad = true;

    if (!this.collection) {
      await this.getCollection();
    }
    if (this.collection) {
      await this.getPhotos();
    }

    if (this.photos.length > 0) {
      this.selectPhoto(this.photos[0]);
    } else {
      this.mightLoad = false;
    }
  }

  private async getCollection() {
    const collectionId = window.location.pathname.split('/').pop();
    if (!collectionId || collectionId.trim() === '') {
      this.router.navigate(['/']);
      return;
    }
    const allCollections = await this.apiService.getCollections();
    this.collection = allCollections.find((c) => c.id === collectionId) || null;
    if (!this.collection) {
      this.router.navigate(['/']);
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
      this.photosLoaded.emit(this.photos);
      this.mightLoad = false;
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
      this.mightLoad = false;
      this.photosLoaded.emit(this.photos);
    } catch (error: any) {
      this.mightLoad = false;
      this.popupService.open(
        'Failed to load photos: ' + error.message,
        5000,
        'error',
      );
      this.photosLoaded.emit([]);
    }
  }
}

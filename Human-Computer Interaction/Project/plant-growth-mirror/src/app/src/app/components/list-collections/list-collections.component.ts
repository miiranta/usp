import {
  Component,
  inject,
  OnInit,
  HostListener,
  Output,
  EventEmitter,
} from '@angular/core';
import { Collection } from '../../models/collection';
import { Photo } from '../../models/photo';
import { ApiService } from '../../services/api.service';
import { PopupService } from '../../services/popup.service';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { SpinnerComponent } from '../spinner/spinner.component';
import { CollectionSelectorComponent } from '../collection-selector/collection-selector.component';
import { firstValueFrom } from 'rxjs';

@Component({
  selector: 'app-list-collections',
  standalone: true,
  imports: [CommonModule, SpinnerComponent, CollectionSelectorComponent],
  templateUrl: './list-collections.component.html',
  styleUrls: ['./list-collections.component.scss'],
})
export class ListCollectionsComponent implements OnInit {
  @Output() createNewCollectionRequested = new EventEmitter<void>();

  private apiService = inject(ApiService);
  private popupService = inject(PopupService);
  private router = inject(Router);
  allCollections: Collection[] = [];
  allPhotos: Photo[] = [];
  photosPerCollection: { [collectionId: string]: Photo[] } = {};
  photosWithoutCollection: Photo[] = [];

  showCollectionSelector: boolean = false;
  selectedPhotoForCollection: Photo | null = null;
  removePhotoPopupOpen: boolean = false;

  mightLoad: boolean = true;
  noPhotosAvailable: boolean = true;

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.removePhotoPopupOpen) {
      if (event.key === 'Enter') {
        event.preventDefault();
        (window as any).res('yes');
      } else if (event.key === 'Escape') {
        event.preventDefault();
        (window as any).res('no');
      }
    }
  }

  ngOnInit(): void {
    this.getAllCollections();
  }

  refreshCollections(): void {
    this.getAllCollections();
  }

  onPhotoCollectionUpdated(): void {
    this.getAllPhotos();
  }

  private async getAllCollections() {
    this.mightLoad = true;

    try {
      this.allCollections = await this.apiService.getCollections();

      this.allCollections.sort((a, b) => {
        const aTime = new Date(a.createdAt).getTime();
        const bTime = new Date(b.createdAt).getTime();
        return bTime - aTime;
      });

      for (const collection of this.allCollections) {
        if (collection.name.length > 30) {
          collection.name = collection.name.substring(0, 27) + '...';
        }
      }

      await this.getAllPhotos();
    } catch (error: any) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.popupService.open(
        'Failed to get collections: ' + errorMessage,
        5000,
        'error',
      );

      this.mightLoad = false;
    }
  }

  private async getAllPhotos() {
    try {
      this.allPhotos = await this.apiService.getPhotos();

      this.photosPerCollection = {};
      this.photosWithoutCollection = [];

      for (const photo of this.allPhotos) {
        if (photo.collectionId) {
          if (!this.photosPerCollection[photo.collectionId]) {
            this.photosPerCollection[photo.collectionId] = [];
          }
          this.photosPerCollection[photo.collectionId].push(photo);
        } else {
          this.photosWithoutCollection.push(photo);
        }
      }

      this.photosWithoutCollection.sort((a, b) => {
        const aTime = new Date(a.createdAt!).getTime();
        const bTime = new Date(b.createdAt!).getTime();
        return bTime - aTime;
      });

      for (const colId of Object.keys(this.photosPerCollection)) {
        this.photosPerCollection[colId].sort((a, b) => {
          const aTime = new Date(a.createdAt!).getTime();
          const bTime = new Date(b.createdAt!).getTime();
          return bTime - aTime;
        });
      }
    } catch (error: any) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.popupService.open(
        'Failed to get photos: ' + errorMessage,
        5000,
        'error',
      );
    } finally {
      this.mightLoad = false;

      if (this.allPhotos.length === 0) {
        this.noPhotosAvailable = true;
      } else {
        this.noPhotosAvailable = false;
      }
    }
  }

  goToPhotoPage(photo: Photo) {
    if (!photo) return;
    this.router.navigate(['/viewPhoto', photo.id]);
  }

  goToCollectionPage(collection: Collection) {
    if (!collection) return;
    this.router.navigate(['/viewCollection', collection.id]);
  }

  isPhotoSelected(photo: Photo): boolean {
    return !!photo.collectionId;
  }

  async onPhotoCheckboxChange(photo: Photo, event: Event) {
    const checkbox = event.target as HTMLInputElement;

    if (!photo.collectionId) {
      if (checkbox.checked) {
        this.selectedPhotoForCollection = photo;
        this.showCollectionSelector = true;
        checkbox.checked = false;
      }
    } else {
      if (!checkbox.checked) {
        await this.removePhotoFromCollection(photo);
        checkbox.checked = true;
      }
    }
  }

  async removePhotoFromCollection(photo: Photo) {
    if (!photo.collectionId) return;

    this.removePhotoPopupOpen = true;
    const popupRes = this.popupService.open(
      `
      <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
        <p>
          Are you sure you want to remove this photo from the collection? <br>
        </p>
        <button class='btn' onclick="res('yes')">Continue</button>
        <button class='btn-info' onclick="res('no')">Cancel</button>
      </div>
      `,
      undefined,
      'error',
    );

    const response = await firstValueFrom(popupRes);
    this.removePhotoPopupOpen = false;
    this.popupService.close();

    if (response === 'no') return;

    try {
      const collection = this.allCollections.find(
        (c) => c.id === photo.collectionId,
      );
      if (collection && collection.photoIds) {
        collection.photoIds = collection.photoIds.filter(
          (id) => id !== photo.id,
        );
        await this.apiService.updateCollection(collection);
      }

      await this.apiService.updatePhotoCollectionId(photo.id, null);

      await this.getAllPhotos();
    } catch (error: any) {
      this.popupService.open(
        'Failed to remove photo from collection: ' + error.message,
        5000,
        'error',
      );
    }
  }
  onCollectionSelected(collection: Collection) {
    if (this.selectedPhotoForCollection) {
      this.getAllCollections();
    }
    this.showCollectionSelector = false;
    this.selectedPhotoForCollection = null;
  }

  onCollectionSelectorCancelled() {
    this.showCollectionSelector = false;
    this.selectedPhotoForCollection = null;
  }

  onCreateNewCollectionRequested() {
    this.showCollectionSelector = false;
    this.createNewCollectionRequested.emit();
  }

  async assignPhotoToNewCollection(newCollection: Collection) {
    if (this.selectedPhotoForCollection && newCollection) {
      try {
        await this.apiService.updatePhotoCollectionId(
          this.selectedPhotoForCollection.id,
          newCollection.id,
        );

        if (!newCollection.photoIds) {
          newCollection.photoIds = [];
        }
        if (
          !newCollection.photoIds.includes(this.selectedPhotoForCollection.id)
        ) {
          newCollection.photoIds.push(this.selectedPhotoForCollection.id);
          await this.apiService.updateCollection(newCollection);
        }

        this.getAllCollections();
      } catch (error: any) {
        this.popupService.open(
          'Failed to add photo to collection: ' + error.message,
          5000,
          'error',
        );
      } finally {
        this.selectedPhotoForCollection = null;
      }
    }
  }
}

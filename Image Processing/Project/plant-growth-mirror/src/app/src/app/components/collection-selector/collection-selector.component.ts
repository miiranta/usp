import {
  Component,
  EventEmitter,
  Input,
  Output,
  inject,
  HostListener,
} from '@angular/core';
import { Collection } from '../../models/collection';
import { Photo } from '../../models/photo';
import { ApiService } from '../../services/api.service';
import { PopupService } from '../../services/popup.service';

@Component({
  selector: 'app-collection-selector',
  standalone: true,
  imports: [],
  templateUrl: './collection-selector.component.html',
  styleUrls: ['./collection-selector.component.scss'],
})
export class CollectionSelectorComponent {
  @Input() collections: Collection[] = [];
  @Input() photo: Photo | null = null;
  @Input() photosPerCollection: { [collectionId: string]: Photo[] } = {};
  @Output() collectionSelected = new EventEmitter<Collection>();
  @Output() cancelled = new EventEmitter<void>();
  @Output() createNewCollectionRequested = new EventEmitter<void>();

  private apiService = inject(ApiService);
  private popupService = inject(PopupService);

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.preventDefault();
      this.cancel();
    }
  }

  getPhotoCount(collectionId: string): number {
    return this.photosPerCollection[collectionId]?.length || 0;
  }

  async selectCollection(collection: Collection) {
    if (!this.photo) return;

    try {
      await this.apiService.updatePhotoCollectionId(
        this.photo.id,
        collection.id,
      );

      if (!collection.photoIds) {
        collection.photoIds = [];
      }
      if (!collection.photoIds.includes(this.photo.id)) {
        collection.photoIds.push(this.photo.id);
        await this.apiService.updateCollection(collection);
      }

      this.collectionSelected.emit(collection);
    } catch (error: any) {
      this.popupService.open(
        'Failed to add photo to collection: ' + error.message,
        5000,
        'error',
      );
    }
  }

  cancel() {
    this.cancelled.emit();
  }

  createNewCollection() {
    this.createNewCollectionRequested.emit();
  }
}

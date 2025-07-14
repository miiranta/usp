import {
  Component,
  inject,
  Input,
  Output,
  HostListener,
  ViewChild,
  ElementRef,
  AfterViewInit,
  OnChanges,
  SimpleChanges,
  EventEmitter,
} from '@angular/core';
import { Collection } from '../../models/collection';
import { ApiService } from '../../services/api.service';
import { Router } from '@angular/router';
import { PopupService } from '../../services/popup.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingService } from '../../services/loading.service';
import { SpinnerComponent } from '../spinner/spinner.component';
import { firstValueFrom } from 'rxjs';

@Component({
  selector: 'app-edit-collection',
  imports: [CommonModule, FormsModule, SpinnerComponent],
  templateUrl: './edit-collection.component.html',
  styleUrl: './edit-collection.component.scss',
})
export class EditCollectionComponent implements OnChanges, AfterViewInit {
  @Input() collection: Collection | null = null;
  @Output() collectionUpdated = new EventEmitter<Collection>();
  @ViewChild('nameInput') nameInput!: ElementRef<HTMLInputElement>;

  private apiService = inject(ApiService);
  private popupService = inject(PopupService);
  private router = inject(Router);
  private loadingService = inject(LoadingService);
  popupOpen = false;

  deleteCollectionPopupOpen = false;
  loadingCollection = false;
  editedName = '';

  constructor() {}

  ngOnChanges(changes: SimpleChanges) {
    if (changes['collection']) {
      if (changes['collection'].currentValue) {
        this.collection = changes['collection'].currentValue;
        this.editedName = this.collection?.name || '';
      } else if (!changes['collection'].firstChange) {
        this.getCollection();
      }
    }
  }

  ngAfterViewInit() {
    if (!this.collection) {
      this.getCollection();
    } else {
      this.editedName = this.collection?.name || '';
    }
  }

  async editCollection(collection: Collection) {
    if (!collection) return;

    this.loadingService.open();

    try {
      const updatedCollection = { ...collection, name: this.editedName };
      const result = await this.apiService.updateCollection(updatedCollection);

      this.collection = result;

      this.collectionUpdated.emit(result);
      this.cancel();
    } catch (error: any) {
      this.popupService.open(
        'Failed to update collection: ' + error.message,
        5000,
        'error',
      );
      return;
    } finally {
      this.loadingService.close();
    }
  }
  async deleteCollection() {
    if (!this.collection) return;

    this.deleteCollectionPopupOpen = true;
    const popupRes = this.popupService.open(
      `
    <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
      <p>
        Are you sure you want to delete the collection? <br>
        All photos will remain in your library.<br>
      </p>
      <button class='btn' onclick="res('yes')">Continue</button>
      <button class='btn-info' onclick="res('no')">Cancel</button>
    </div>
    `,
      undefined,
      'error',
    );
    const response = await firstValueFrom(popupRes);
    this.deleteCollectionPopupOpen = false;
    this.popupService.close();
    if (response === 'no') return;

    this.loadingService.open();
    try {
      await this.apiService.deleteCollection(this.collection.id);
    } catch (error: any) {
      this.popupService.open(
        'Failed to delete collection: ' + error.message,
        5000,
        'error',
      );
    } finally {
      this.router.navigate(['/']);
    }
  }
  cancel() {
    this.popupOpen = false;
    this.editedName = this.collection?.name || '';
  }
  openEditPopup() {
    this.editedName = this.collection?.name || '';
    this.popupOpen = true;

    setTimeout(() => {
      if (this.nameInput?.nativeElement) {
        this.nameInput.nativeElement.focus();
        this.nameInput.nativeElement.select();
      }
    }, 0);
  }

  onInputKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && this.editedName.trim() && this.collection) {
      event.preventDefault();
      this.editCollection(this.collection);
    }
  }
  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.deleteCollectionPopupOpen) {
      if (event.key === 'Enter') {
        event.preventDefault();
        (window as any).res('yes');
      } else if (event.key === 'Escape') {
        event.preventDefault();
        (window as any).res('no');
      }
    } else if (event.key === 'Escape' && this.popupOpen) {
      this.cancel();
    }
  }

  private async getCollection() {
    const collectionId = window.location.pathname.split('/').pop();
    if (!collectionId || collectionId.trim() === '') {
      this.router.navigate(['/']);
      return;
    }
    this.loadingCollection = true;
    const allCollections = await this.apiService.getCollections();
    this.collection = allCollections.find((c) => c.id === collectionId) || null;
    if (!this.collection) {
      this.router.navigate(['/']);
      return;
    }
    this.loadingCollection = false;
  }

  getCollectionName(): string | null {
    let collectionName = this.collection?.name;

    if (collectionName && collectionName.length > 30) {
      collectionName = collectionName.substring(0, 27) + '...';
    }

    return collectionName || null;
  }
}

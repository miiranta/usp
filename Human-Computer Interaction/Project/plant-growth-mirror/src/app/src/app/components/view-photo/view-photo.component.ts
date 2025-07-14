import {
  Component,
  EventEmitter,
  inject,
  Input,
  Output,
  HostListener,
} from '@angular/core';
import { EditService } from '../../services/edit.service';
import { Photo } from '../../models/photo';
import { ApiService } from '../../services/api.service';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { SpinnerComponent } from '../spinner/spinner.component';
import { firstValueFrom } from 'rxjs';
import { PopupService } from '../../services/popup.service';

@Component({
  selector: 'app-view-photo',
  imports: [CommonModule, SpinnerComponent],
  templateUrl: './view-photo.component.html',
  styleUrl: './view-photo.component.scss',
})
export class ViewPhotoComponent {
  @Input() photo: Photo | null = null;
  @Input() allowRedirect: boolean = true;

  @Output() refreshListing = new EventEmitter<void>();

  private apiService = inject(ApiService);
  private editService = inject(EditService);
  private popupService = inject(PopupService);
  private router = inject(Router);
  collectionName: string | null = null;
  enableLoadingSpinner: boolean = true;
  deleting: boolean = false;
  deletionPopupOpen: boolean = false;

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.deletionPopupOpen) {
      if (event.key === 'Enter') {
        event.preventDefault();
        (window as any).res('yes');
      } else if (event.key === 'Escape') {
        event.preventDefault();
        (window as any).res('no');
      }
    }
  }

  ngOnInit() {
    if (!this.photo) {
      this.getPhoto();
    } else {
      this.getCollectionName();
    }
  }

  onEditPhoto() {
    if (!this.photo) return;
    this.editService.clearStagedPhotos();
    this.editService.pushStagedPhoto(this.photo);
    this.editService.setPreviousRoute(this.router.url);
    this.router.navigate(['/editPhoto']);
  }
  async onDeletePhoto() {
    if (!this.photo) {
      return;
    }

    if (this.deleting) {
      return;
    }

    this.deletionPopupOpen = true;
    const popupRes = this.popupService.open(
      `
    <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
      <p>
        Are you sure you want to delete this photo? <br>
      </p>
      <button class='btn' onclick="res('yes')">Continue</button>
      <button class='btn-info' onclick="res('no')">Cancel</button>
    </div>
    `,
      undefined,
      'error',
    );
    const response = await firstValueFrom(popupRes);
    this.deletionPopupOpen = false;
    this.popupService.close();
    if (response === 'no') return;

    this.deleting = true;

    try {
      await this.apiService.deletePhoto(this.photo.id);
      this.refreshListing.emit();
      if (this.allowRedirect) this.router.navigate(['/']);
    } catch (error) {
      this.popupService.open(
        `<div style="text-align: center;">
          <p>Failed to delete photo. Please try again.</p>
          <p style="color: red; font-size: 0.9em;">${error instanceof Error ? error.message : 'Unknown error'}</p>
          <button class='btn' onclick="res('ok')">OK</button>
        </div>`,
        undefined,
        'error',
      );
    } finally {
      this.deleting = false;
    }
  }

  goToCollection(collectionId: string) {
    if (!collectionId) return;
    this.router.navigate(['/viewCollection', collectionId]);
  }

  private async getPhoto() {
    const photoId = window.location.pathname.split('/').pop();
    if (!photoId || photoId.trim() === '') {
      if (!this.allowRedirect) {
        this.enableLoadingSpinner = false;
        return;
      }
      this.router.navigate(['/']);
      return;
    }
    this.photo = await this.apiService.getPhoto(photoId);
    if (!this.photo) {
      if (!this.allowRedirect) {
        this.enableLoadingSpinner = false;
        return;
      }
      this.router.navigate(['/']);
      return;
    }

    await this.getCollectionName();
  }

  private async getCollectionName() {
    if (!this.photo) return;
    const allCollections = await this.apiService.getCollections();
    const collection = this.photo
      ? allCollections.find((c) => c.id === this.photo!.collectionId)
      : null;
    this.collectionName = collection ? collection.name : null;

    if (this.collectionName && this.collectionName.length > 30) {
      this.collectionName = this.collectionName.substring(0, 27) + '...';
    }
  }
}

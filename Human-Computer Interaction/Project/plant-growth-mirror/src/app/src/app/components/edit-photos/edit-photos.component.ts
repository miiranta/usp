import {
  Component,
  inject,
  Input,
  Output,
  EventEmitter,
  OnInit,
  OnChanges,
  SimpleChanges,
  HostListener,
  ViewChild,
  ViewChildren,
  QueryList,
} from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { EditService } from '../../services/edit.service';
import { ApiService } from '../../services/api.service';
import { PopupService } from '../../services/popup.service';
import { LoadingService } from '../../services/loading.service';
import { Photo } from '../../models/photo';
import { DateUtils } from '../../utils/date';
import { first, firstValueFrom } from 'rxjs';
import { Collection } from '../../models/collection';
import { SpinnerComponent } from '../spinner/spinner.component';

@Component({
  selector: 'app-edit-photos',
  standalone: true,
  imports: [CommonModule, FormsModule, SpinnerComponent],
  templateUrl: './edit-photos.component.html',
  styleUrls: ['./edit-photos.component.scss'],
})
export class EditPhotosComponent implements OnInit {
  private editService = inject(EditService);
  private apiService = inject(ApiService);
  private popupService = inject(PopupService);
  private loadingService = inject(LoadingService);
  private router = inject(Router);

  @Input() collectionId: string | null = null;
  @ViewChild('controls') controls: any;
  @ViewChildren(SpinnerComponent) spinners!: QueryList<SpinnerComponent>;
  collection: Collection | null = null;
  photos: Photo[] = [];
  doneConfirmationPopupOpen: boolean = false;

  photoParams: {
    [photoId: string]: {
      granularity: number;
      threshold: number;
      createdAt: string;
    };
  } = {};
  photoParamsAutoeditClone: {
    [photoId: string]: {
      granularity: number;
      threshold: number;
      createdAt: string;
    };
  } = {};
  autoedit = false;
  processAllMode = false;

  globalGranularity = this.editService.getGranularity();
  globalThreshold = this.editService.getThreshold();

  photoBackups: {
    [photoId: string]: Photo[];
  } = {};

  photoRedos: {
    [photoId: string]: Photo[];
  } = {};

  photoLoading: { [photoId: string]: boolean } = {};
  currentIndex = 0;
  selectedPhotoId: string | null = null;
  globalOperationInProgress = false;

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.doneConfirmationPopupOpen) {
      if (event.key === 'Enter') {
        event.preventDefault();
        (window as any).res('yes');
      } else if (event.key === 'Escape') {
        event.preventDefault();
        (window as any).res('no');
      }
    }
  }

  constructor() {
    this.photos = this.editService.getStagedPhotos();

    if (!this.photos.length) {
      this.router.navigate(['/']);
      return;
    } else {
      this.editService.setCollectionId(this.photos[0].collectionId ?? null);
    }
  }

  ngOnInit(): void {
    this.initializeParams();
    this.sortPhotos();

    if (!this.selectedPhotoId && this.photos.length > 0) {
      this.selectedPhotoId = this.photos[0].id;
      this.currentIndex = 0;
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (
      changes['collectionId'] &&
      changes['collectionId'].currentValue !==
        changes['collectionId'].previousValue
    ) {
      this.onCollectionUpdate();
    }
  }

  ngAfterViewInit() {
    this.copyPhotoParamsToAutoeditClone();

    this.controls.nativeElement.addEventListener(
      'click',
      (event: MouseEvent) => {
        this.onParamsChange();
      },
    );
  }

  async onCollectionUpdate() {
    this.photos.forEach((photo) => {
      photo.collectionId = this.collectionId;
      this.updatePhotoCollectionId(photo);
    });

    await this.getCollection();
    if (this.collection) {
      const editedPhotoIds = this.photos.map((p) => p.id);
      const existingPhotoIds = this.collection.photoIds ?? [];
      const preservedPhotoIds = existingPhotoIds.filter(
        (id) => !editedPhotoIds.includes(id),
      );
      this.collection.photoIds = [...preservedPhotoIds, ...editedPhotoIds];
      await this.updateCollection();
    }

    await Promise.all(
      this.photos.map((photo) => {
        return this.apiService.createOrEditPhotoNoProc(photo);
      }),
    );
  }

  onParamsChange() {
    if (!this.autoedit) return;

    const isIdentical =
      JSON.stringify(this.photoParams) ===
      JSON.stringify(this.photoParamsAutoeditClone);
    if (!isIdentical) this.editPhoto(this.photos[this.currentIndex]);
    this.copyPhotoParamsToAutoeditClone();
  }

  private sortPhotos() {
    this.photos.sort(
      (a, b) =>
        new Date(a.createdAt ?? 0).getTime() -
        new Date(b.createdAt ?? 0).getTime(),
    );

    this.updateIndex();
  }
  private initializeParams() {
    this.photos.forEach((photo) => {
      this.copyPhotoParamsToForm(photo);
      this.photoBackups[photo.id] = [];
      this.photoRedos[photo.id] = [];
      this.photoLoading[photo.id] = false;
    });

    this.updateIndex();
  }

  async editPhoto(photo: Photo) {
    const id = photo.id;

    const dateInput = this.photoParams[id].createdAt;
    if (!this.isValidDateInput(dateInput)) {
      this.popupService.open(
        'Please enter a complete date and time (e.g., 2025-06-13 14:30).',
        5000,
        'error',
      );
      return;
    }

    this.photoLoading[id] = true;

    this.photoBackups[id].push({ ...photo });

    this.photoRedos[id] = [];

    photo.params = {
      granularity: this.photoParams[id].granularity,
      threshold: this.photoParams[id].threshold,
    };
    photo.createdAt = DateUtils.inputToDate(this.photoParams[id].createdAt);

    try {
      photo = await this.apiService.createOrEditPhoto(photo);
      this.photos = this.photos.map((p) => (p.id === id ? { ...photo } : p));

      this.completeAllSpinners();
    } catch (error: any) {
      this.popupService.open(
        `Error editing photo: ${error.message}`,
        5000,
        'error',
      );
    } finally {
      this.photoLoading[id] = false;
      this.sortPhotos();
    }
  }
  undoPhoto(photo: Photo) {
    const id = photo.id;

    if (this.photoBackups[id].length === 0 && !this.globalOperationInProgress) {
      this.popupService.open('Nothing to undo for this photo.', 2000, 'info');
      return;
    }

    this.photoRedos[id].push({ ...photo });

    const lastBackup = this.photoBackups[id].pop();

    if (lastBackup) {
      this.photos = this.photos.map((p) =>
        p.id === id ? { ...lastBackup } : p,
      );
      this.copyPhotoParamsToForm(lastBackup);
    }

    this.copyPhotoParamsToAutoeditClone();
    this.sortPhotos();

    this.apiService.createOrEditPhotoNoProc(lastBackup!).catch((error) => {
      this.popupService.open(
        `Error undoing photo: ${error.message}`,
        5000,
        'error',
      );
    });
  }

  redoPhoto(photo: Photo) {
    const id = photo.id;

    if (this.photoRedos[id].length === 0 && !this.globalOperationInProgress) {
      this.popupService.open('Nothing to redo for this photo.', 2000, 'info');
      return;
    }

    this.photoBackups[id].push({ ...photo });

    const nextRedo = this.photoRedos[id].pop();

    if (nextRedo) {
      this.photos = this.photos.map((p) => (p.id === id ? { ...nextRedo } : p));
      this.copyPhotoParamsToForm(nextRedo);
    }

    this.copyPhotoParamsToAutoeditClone();
    this.sortPhotos();

    this.apiService.createOrEditPhotoNoProc(nextRedo!).catch((error) => {
      this.popupService.open(
        `Error redoing photo: ${error.message}`,
        5000,
        'error',
      );
    });
  }

  async editAll() {
    this.globalOperationInProgress = true;

    this.photos.forEach((photo) => {
      this.photoParams[photo.id] = {
        granularity: this.globalGranularity,
        threshold: this.globalThreshold,
        createdAt: DateUtils.dateToInput(photo.createdAt ?? new Date()),
      };
    });

    this.editService.setGranularity(this.globalGranularity);
    this.editService.setThreshold(this.globalThreshold);

    await Promise.all(
      this.photos.map((photo) => {
        return this.editPhoto(photo);
      }),
    );

    this.completeAllSpinners();

    this.globalOperationInProgress = false;
  }

  undoAll() {
    this.globalOperationInProgress = true;

    this.photos.forEach((photo) => {
      this.undoPhoto(photo);
    });

    this.globalOperationInProgress = false;
  }

  redoAll() {
    this.globalOperationInProgress = true;

    this.photos.forEach((photo) => {
      this.redoPhoto(photo);
    });

    this.globalOperationInProgress = false;
  }
  async done() {
    if (!this.arePhotosAllProcessed()) {
      this.doneConfirmationPopupOpen = true;
      const popupRes = this.popupService.open(
        `
      <div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
        <p>
          Not all photos are processed. <br> 
          Do you want to continue anyway? <br>
        </p>
        <button class='btn' onclick="res('yes')">Continue</button>
        <button class='btn-info' onclick="res('no')">Cancel</button>
      </div>
      `,
        undefined,
        'info',
      );
      const response = await firstValueFrom(popupRes);
      this.doneConfirmationPopupOpen = false;
      this.popupService.close();
      if (response === 'no') return;
    }

    this.loadingService.open();

    this.onCollectionUpdate();

    const hasCollection =
      this.collectionId !== null &&
      this.photos.every((p) => p.collectionId === this.collectionId);
    if (hasCollection) {
      this.router.navigate(['/viewCollection', this.collectionId]);
      return;
    }

    if (this.photos.length === 1) {
      this.router.navigate(['/viewPhoto', this.photos[0].id]);
      return;
    }

    this.router.navigate(['/']);
  }

  isPhotoProcessed(photo: Photo): boolean {
    return !!photo.dataProcessedBase64 && photo.dataProcessedBase64.length > 0;
  }

  arePhotosAllProcessed(): boolean {
    return this.photos.every((photo) => this.isPhotoProcessed(photo));
  }

  selectThumbnail(idx: number) {
    this.currentIndex = idx;

    const selectedPhoto = this.photos[idx];
    this.selectedPhotoId = selectedPhoto.id;
  }

  private updateIndex() {
    if (this.selectedPhotoId) {
      const newIndex = this.photos.findIndex(
        (p) => p.id === this.selectedPhotoId,
      );
      this.currentIndex = newIndex >= 0 ? newIndex : 0;
    }

    if (this.currentIndex >= this.photos.length) {
      this.currentIndex = this.photos.length - 1;
    }
  }

  private copyPhotoParamsToForm(photo: Photo) {
    this.photoParams[photo.id] = {
      granularity: photo.params?.granularity ?? this.globalGranularity,
      threshold: photo.params?.threshold ?? this.globalThreshold,
      createdAt: photo.createdAt
        ? DateUtils.dateToInput(photo.createdAt)
        : DateUtils.dateToInput(new Date()),
    };
  }

  private copyPhotoParamsToAutoeditClone() {
    this.photoParamsAutoeditClone = JSON.parse(
      JSON.stringify(this.photoParams),
    );
  }

  private async getCollection() {
    const collectionId = this.collectionId;
    if (!collectionId || collectionId.trim() === '' || collectionId === null) {
      return;
    }
    const allCollections = await this.apiService.getCollections();
    this.collection = allCollections.find((c) => c.id === collectionId) || null;
    if (!this.collection) {
      return;
    }
  }

  private async updatePhotoCollectionId(photo: Photo) {
    try {
      await this.apiService.updatePhotoCollectionId(
        photo.id,
        photo.collectionId ?? null,
      );
    } catch (error: any) {
      this.popupService.open(
        `Error updating photo collection ID: ${error.message}`,
        5000,
        'error',
      );
    }
  }
  private async updateCollection() {
    if (this.collection) {
      await this.apiService.updateCollection(this.collection);
    }
  }

  private isValidDateInput(dateInput: string): boolean {
    if (!dateInput || typeof dateInput !== 'string') {
      return false;
    }

    const dateTimePattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
    if (!dateTimePattern.test(dateInput)) {
      return false;
    }

    const date = new Date(dateInput);
    if (isNaN(date.getTime())) {
      return false;
    }

    const formattedBack = DateUtils.dateToInput(date);
    return formattedBack === dateInput;
  }

  private completeAllSpinners() {
    if (this.spinners) {
      this.spinners.forEach((spinner) => {
        spinner.completeProgress();
      });
    }
  }
}

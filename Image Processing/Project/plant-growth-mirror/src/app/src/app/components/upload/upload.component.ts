import {
  Component,
  Output,
  EventEmitter,
  inject,
  HostListener,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { PopupService } from '../../services/popup.service';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
})
export class UploadComponent {
  @Output() upload = new EventEmitter<File[]>();

  private popupService = inject(PopupService);

  files: File[] = [];
  isDragOver = false;

  showDuplicateConfirmation = false;
  duplicateFileNames: string[] = [];
  pendingFiles: File[] = [];

  acceptedFormats = [
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/gif',
    'image/webp',
    'image/tiff',
    'image/tif',
    'image/bmp',
  ];

  acceptedExtensions = [
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.webp',
    '.tiff',
    '.tif',
    '.bmp',
  ];

  constructor() {}

  onFilesSelected(event: Event) {
    const inputEl = event.target as HTMLInputElement;
    if (!inputEl.files) return;

    this.addFiles(Array.from(inputEl.files));
    inputEl.value = '';
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragOver = false;

    if (event.dataTransfer?.files) {
      this.addFiles(Array.from(event.dataTransfer.files));
    }
  }
  private addFiles(newFiles: File[]) {
    const validFiles: File[] = [];
    const duplicateFiles: File[] = [];
    const invalidFiles: string[] = [];

    newFiles.forEach((file) => {
      if (!this.isValidImageFile(file)) {
        invalidFiles.push(file.name);
        return;
      }

      const exists = this.files.some(
        (existing) =>
          existing.name === file.name && existing.size === file.size,
      );

      if (!exists) {
        validFiles.push(file);
      } else {
        duplicateFiles.push(file);
      }
    });

    this.files.push(...validFiles);

    if (duplicateFiles.length > 0) {
      this.pendingFiles = duplicateFiles;
      this.duplicateFileNames = duplicateFiles.map((file) => file.name);
      this.showDuplicateConfirmation = true;
    }

    if (invalidFiles.length > 0) {
      const fileList = invalidFiles.map((name) => `• ${name}`).join('<br>');
      const acceptedList = this.acceptedExtensions.join(', ');
      this.popupService.open(
        `<div style="text-align: left;">
          <strong>Files with invalid formats:</strong><br>
          ${fileList}<br><br>
          <strong>Accepted formats:</strong><br>
          ${acceptedList}
        </div>`,
        undefined,
        'error',
      );
    }
  }

  private isValidImageFile(file: File): boolean {
    if (this.acceptedFormats.includes(file.type)) {
      return true;
    }

    const fileName = file.name.toLowerCase();
    return this.acceptedExtensions.some((ext) => fileName.endsWith(ext));
  }

  removeFile(index: number) {
    this.files.splice(index, 1);
  }

  uploadFiles() {
    if (this.files.length === 0) return;

    this.upload.emit(this.files.slice());
  }

  clearFiles() {
    this.files = [];
  }

  confirmAddDuplicates() {
    this.files.push(...this.pendingFiles);
    this.closeDuplicateConfirmation();
  }

  cancelDuplicateConfirmation() {
    this.closeDuplicateConfirmation();
  }
  private closeDuplicateConfirmation() {
    this.showDuplicateConfirmation = false;
    this.duplicateFileNames = [];
    this.pendingFiles = [];
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (this.showDuplicateConfirmation) {
      if (event.key === 'Enter') {
        event.preventDefault();
        this.confirmAddDuplicates();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        this.cancelDuplicateConfirmation();
      }
    }
  }
}

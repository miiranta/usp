import { Component, inject, ViewChild } from '@angular/core';
import { UploadComponent } from '../../components/upload/upload.component';
import { ManageCollectionsComponent } from '../../components/manage-collections/manage-collections.component';
import { Photo } from '../../models/photo';
import { Collection } from '../../models/collection';
import { EditService } from '../../services/edit.service';
import { Router } from '@angular/router';
import { FileConversion } from '../../utils/fileConversion';
import { ApiService } from '../../services/api.service';
import { LoadingService } from '../../services/loading.service';
import { ListCollectionsComponent } from '../../components/list-collections/list-collections.component';

@Component({
  selector: 'app-home-page',
  imports: [
    UploadComponent,
    ManageCollectionsComponent,
    ListCollectionsComponent,
  ],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.scss',
})
export class HomePageComponent {
  @ViewChild(ListCollectionsComponent)
  listCollectionsComponent!: ListCollectionsComponent;
  @ViewChild(UploadComponent) uploadComponent!: UploadComponent;
  @ViewChild(ManageCollectionsComponent)
  manageCollectionsComponent!: ManageCollectionsComponent;

  private editService = inject(EditService);
  private apiService = inject(ApiService);
  private loadingService = inject(LoadingService);
  private router = inject(Router);

  private selectedCollection: Collection | null = null;

  ngAfterViewInit() {
    this.loadingService.close();
  }
  async onUpload(files: File[]) {
    this.loadingService.open(`Uploading ${files.length} files...`);
    this.editService.clearStagedPhotos();

    let uploadedCount = 0;
    const totalFiles = files.length;

    const promises = files.map(async (file, index) => {
      const base64 = await FileConversion.fileToBase64(file);

      let photo: Photo = {
        id: '',
        collectionId: this.selectedCollection
          ? this.selectedCollection.id
          : null,
        dataBase64: base64,
        timestamp:
          new Date(file.lastModified).toISOString() || new Date().toISOString(),
      };

      const photoRes = await this.apiService.createOrEditPhoto(photo);
      uploadedCount++;

      const remaining = totalFiles - uploadedCount;
      if (remaining > 0) {
        this.loadingService.updateText(
          `${remaining} files remaining to upload...`,
        );
      } else {
        this.loadingService.updateText('Upload complete!');
      }

      if (this.selectedCollection) {
        (this.selectedCollection.photoIds ??= []).push(photoRes.id);
        await this.apiService.updateCollection(this.selectedCollection);
      }
    });
    await Promise.all(promises);

    setTimeout(() => {
      if (this.listCollectionsComponent) {
        this.listCollectionsComponent.refreshCollections();
      }

      if (this.uploadComponent) {
        this.uploadComponent.clearFiles();
      }

      this.loadingService.close();
    }, 1000);
  }

  onCollectionSelectionChange(collection: Collection | null): void {
    this.selectedCollection = collection;
  }
  onCollectionCreated(collection: Collection): void {
    if (this.listCollectionsComponent) {
      this.listCollectionsComponent.refreshCollections();
      this.listCollectionsComponent.assignPhotoToNewCollection(collection);
    }
  }

  onCreateNewCollectionRequested(): void {
    if (this.manageCollectionsComponent) {
      this.manageCollectionsComponent.openPopup();
    }
  }
}

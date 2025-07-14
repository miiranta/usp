import { Component, inject } from '@angular/core';
import { EditPhotosComponent } from '../../components/edit-photos/edit-photos.component';
import { Collection } from '../../models/collection';
import { ManageCollectionsComponent } from '../../components/manage-collections/manage-collections.component';
import { LoadingService } from '../../services/loading.service';

@Component({
  selector: 'app-edit-photo-page',
  imports: [EditPhotosComponent, ManageCollectionsComponent],
  templateUrl: './edit-photo-page.component.html',
  styleUrl: './edit-photo-page.component.scss',
})
export class EditPhotoPageComponent {
  private loadingService = inject(LoadingService);
  collectionId: string | null = null;

  ngAfterViewInit() {
    this.loadingService.close();
  }

  onCollectionSelectionChange(collection: Collection | null): void {
    if (collection) {
      this.collectionId = collection.id;
    } else {
      this.collectionId = null;
    }
  }
}

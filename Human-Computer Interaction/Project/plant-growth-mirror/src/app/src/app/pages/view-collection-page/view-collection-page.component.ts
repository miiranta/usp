import { Component, inject, OnInit, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { LoadingService } from '../../services/loading.service';
import { ApiService } from '../../services/api.service';
import { EditService } from '../../services/edit.service';
import { PopupService } from '../../services/popup.service';
import { Collection } from '../../models/collection';
import { Photo } from '../../models/photo';
import { EditCollectionComponent } from '../../components/edit-collection/edit-collection.component';
import { ListCollectionPhotosComponent } from '../../components/list-collection-photos/list-collection-photos.component';
import { ShowGraphsComponent } from '../../components/show-graphs/show-graphs.component';
import { ManageCollectionsComponent } from '../../components/manage-collections/manage-collections.component';

@Component({
  selector: 'app-view-collection-page',
  imports: [
    EditCollectionComponent,
    ListCollectionPhotosComponent,
    ShowGraphsComponent,
    ManageCollectionsComponent,
  ],
  templateUrl: './view-collection-page.component.html',
  styleUrl: './view-collection-page.component.scss',
})
export class ViewCollectionPageComponent implements OnInit {
  @ViewChild('manageCollections')
  manageCollections!: ManageCollectionsComponent;

  private loadingService = inject(LoadingService);
  private apiService = inject(ApiService);
  private editService = inject(EditService);
  private popupService = inject(PopupService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);

  currentCollection: Collection | null = null;
  collectionId: string | null = null;
  collectionPhotos: Photo[] = [];

  ngOnInit() {
    this.route.params.subscribe((params) => {
      this.collectionId = params['id'] || null;
      this.editService.setCollectionId(this.collectionId);
      this.loadCurrentCollection();
    });
  }

  ngAfterViewInit() {
    this.loadingService.close();
  }

  async loadCurrentCollection() {
    if (!this.collectionId) {
      this.currentCollection = null;
      return;
    }

    try {
      const collections = await this.apiService.getCollections();
      this.currentCollection =
        collections.find((c) => c.id === this.collectionId) || null;

      if (!this.currentCollection) {
        this.popupService.open(`Collection not found`, 5000, 'error');
      }
    } catch (error) {
      this.popupService.open(
        `Error loading collection: ${(error as Error).message}`,
        5000,
        'error',
      );
      this.currentCollection = null;
    }
  }

  onCollectionChanged(collection: Collection | null) {
    this.currentCollection = collection;
    this.collectionId = collection?.id || null;
    this.collectionPhotos = [];
    this.editService.setCollectionId(this.collectionId);

    if (collection) {
      this.router.navigate(['/viewCollection', collection.id], {
        replaceUrl: true,
      });
    } else {
      this.router.navigate(['/'], { replaceUrl: true });
    }
  }

  onCollectionUpdated(updatedCollection: Collection) {
    this.currentCollection = updatedCollection;
    if (this.manageCollections) {
      this.manageCollections.refreshCollection(updatedCollection);
    }
  }

  onPhotosLoaded(photos: Photo[]) {
    this.collectionPhotos = photos;
  }
}

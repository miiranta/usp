import {
  Component,
  inject,
  Output,
  EventEmitter,
  OnInit,
  HostListener,
  ViewChild,
  ElementRef,
  Input,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { PopupService } from '../../services/popup.service';
import { Collection } from '../../models/collection';
import { Router } from '@angular/router';
import { LoadingService } from '../../services/loading.service';
import { EditService } from '../../services/edit.service';
import { SpinnerComponent } from '../spinner/spinner.component';

@Component({
  selector: 'app-manage-collections',
  standalone: true,
  imports: [CommonModule, FormsModule, SpinnerComponent],
  templateUrl: './manage-collections.component.html',
  styleUrls: ['./manage-collections.component.scss'],
})
export class ManageCollectionsComponent implements OnInit {
  @Input() initialCollectionId: string | null = null;
  @Input() hideRedirectButton: boolean = false;
  @Output() selectionChange = new EventEmitter<Collection | null>();
  @Output() collectionCreated = new EventEmitter<Collection>();
  @ViewChild('newCollectionInput')
  newCollectionInput!: ElementRef<HTMLInputElement>;

  private apiService = inject(ApiService);
  private popupService = inject(PopupService);
  private loadingService = inject(LoadingService);
  private editService = inject(EditService);
  private router = inject(Router);

  collections: Collection[] = [];
  selectedId: string | null = null;

  showPopup = false;
  newCollectionName = '';
  dropdownOpen = false;
  loadingCollections = false;

  ngOnInit() {
    this.loadCollections();
  }

  ngAfterViewInit() {
    this.selectedId =
      this.initialCollectionId || this.editService.getCollectionId();
  }

  async loadCollections() {
    try {
      this.loadingCollections = true;
      this.collections = await this.apiService.getCollections();
      if (this.initialCollectionId && !this.selectedId) {
        this.selectedId = this.initialCollectionId;
      }
      this.emitSelection();
      this.loadingCollections = false;
    } catch (error) {
      this.popupService.open(
        `Error loading collections: ${(error as Error).message}`,
        5000,
        'error',
      );
      this.collections = [];
    }
  }

  toggleDropdown() {
    this.dropdownOpen = !this.dropdownOpen;
  }

  closeDropdown() {
    this.dropdownOpen = false;
  }

  async selectCollection(id: string | null, event?: MouseEvent) {
    if (event) event.stopPropagation();

    if (this.selectedId === id) {
      this.closeDropdown();
      return;
    }

    this.selectedId = id;
    this.closeDropdown();

    this.emitSelection();
  }

  emitSelection() {
    const selected =
      this.collections.find((c) => c.id === this.selectedId) || null;
    this.editService.setCollectionId(this.selectedId);
    this.selectionChange.emit(selected);
  }

  selectedCollectionName(): string {
    let c = this.collections.find((c) => c.id === this.selectedId);

    if (c && c.name && c.name.length > 30) {
      c.name = c.name.substring(0, 27) + '...';
    }

    return c?.name || 'Click to select a collection...';
  }

  openPopup() {
    this.newCollectionName = '';
    this.showPopup = true;

    setTimeout(() => {
      if (this.newCollectionInput?.nativeElement) {
        this.newCollectionInput.nativeElement.focus();
        this.newCollectionInput.nativeElement.select();
      }
    }, 0);
  }

  closePopup() {
    this.showPopup = false;
  }

  onInputKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && this.newCollectionName.trim()) {
      event.preventDefault();
      this.createAndSelect();
    }
  }

  async createAndSelect() {
    if (!this.newCollectionName.trim()) return;
    try {
      this.loadingService.open();
      const created = await this.apiService.createCollection(
        this.newCollectionName.trim(),
      );
      await this.loadCollections();
      this.selectedId = created.id;
      this.emitSelection();
      this.collectionCreated.emit(created);
      this.closePopup();
      this.popupService.open(
        `Collection "${created.name}" created successfully!`,
        5000,
        'success',
      );
    } catch (error) {
      this.popupService.open(
        `Error creating collection: ${(error as Error).message}`,
        5000,
        'error',
      );
    } finally {
      this.loadingService.close();
    }
  }

  openDetails(collectionId: string, event: MouseEvent) {
    event.stopPropagation();
    this.router.navigate(['/viewCollection', collectionId]);
  }

  refreshCollection(updatedCollection: Collection) {
    const index = this.collections.findIndex(
      (c) => c.id === updatedCollection.id,
    );
    if (index !== -1) {
      this.collections[index] = updatedCollection;
    }
  }

  @HostListener('document:click', ['$event'])
  onGlobalClick(event: MouseEvent) {
    const target = event.target as HTMLElement;
    const clickedInside = target.closest('.custom-dropdown');
    if (!clickedInside) {
      this.closeDropdown();
    }
  }
  @HostListener('document:keydown', ['$event'])
  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      if (this.showPopup) {
        this.closePopup();
      } else if (this.dropdownOpen) {
        this.closeDropdown();
      }
    }
  }

  @HostListener('window:scroll', ['$event'])
  onWindowScroll() {
    if (this.dropdownOpen) {
      this.closeDropdown();
    }
  }

  @HostListener('window:resize', ['$event'])
  onWindowResize() {
    if (this.dropdownOpen) {
      this.closeDropdown();
    }
  }
}

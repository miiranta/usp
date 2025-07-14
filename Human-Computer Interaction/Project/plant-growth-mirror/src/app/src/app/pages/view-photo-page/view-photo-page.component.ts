import { Component, inject } from '@angular/core';
import { LoadingService } from '../../services/loading.service';
import { ViewPhotoComponent } from '../../components/view-photo/view-photo.component';

@Component({
  selector: 'app-view-photo-page',
  imports: [ViewPhotoComponent],
  templateUrl: './view-photo-page.component.html',
  styleUrl: './view-photo-page.component.scss',
})
export class ViewPhotoPageComponent {
  private loadingService = inject(LoadingService);

  ngAfterViewInit() {
    this.loadingService.close();
  }
}

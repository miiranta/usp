import { Routes } from '@angular/router';
import { HomePageComponent } from './pages/home-page/home-page.component';
import { ViewPhotoPageComponent } from './pages/view-photo-page/view-photo-page.component';
import { EditPhotoPageComponent } from './pages/edit-photo-page/edit-photo-page.component';
import { ViewCollectionPageComponent } from './pages/view-collection-page/view-collection-page.component';

export const routes: Routes = [
  { path: '', component: HomePageComponent },
  { path: 'viewPhoto/:id', component: ViewPhotoPageComponent },
  { path: 'viewCollection/:id', component: ViewCollectionPageComponent },
  { path: 'editPhoto', component: EditPhotoPageComponent },

  { path: '**', redirectTo: '' },
];

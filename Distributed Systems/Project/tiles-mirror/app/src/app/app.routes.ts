import { Routes } from '@angular/router';
import { HomePageComponent } from './pages/home-page/home-page.component';
import { GamePageComponent } from './pages/game-page/game-page.component';

export const routes: Routes = [
  { path: '', component: HomePageComponent },
  { path: 'tiles', component: GamePageComponent },
  { path: '**', redirectTo: '' },
];

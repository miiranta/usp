import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LoadingComponent } from './components/loading/loading.component';
import { PopupComponent } from './components/popup/popup.component';
import { TitlebarComponent } from './components/titlebar/titlebar.component';
import { ScrollableContainerComponent } from './components/scrollable-container/scrollable-container.component';

@Component({
  selector: 'app-root',
  imports: [
    RouterOutlet,
    LoadingComponent,
    PopupComponent,
    TitlebarComponent,
    ScrollableContainerComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'plant-growth';
}

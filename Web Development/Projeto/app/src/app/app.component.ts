import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms'; // Import FormsModule

import { TilesComponent } from './tiles/tiles.component';

@Component({
  selector: 'app-root',
  imports: [TilesComponent, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  name: string = '';
  welcomeScreen: boolean = true;

  openGame() {
    this.welcomeScreen = false;
  }
  
}

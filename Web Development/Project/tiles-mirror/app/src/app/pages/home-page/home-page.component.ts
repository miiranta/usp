import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChooseNameComponent } from '../../components/choose-name/choose-name.component';
import { PasswordComponent } from '../../components/password/password.component';

@Component({
  selector: 'app-home-page',
  imports: [CommonModule, ChooseNameComponent, PasswordComponent],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.scss',
})
export class HomePageComponent {
  showPasswordComponent: boolean = false;
  playerName: string = '';
  isNewPlayer: boolean = false;

  onPasswordNeeded(data: { playerName: string; isNewPlayer: boolean }) {
    this.playerName = data.playerName;
    this.isNewPlayer = data.isNewPlayer;
    this.showPasswordComponent = true;
  }

  onBackToChooseName() {
    this.showPasswordComponent = false;
    this.playerName = '';
    this.isNewPlayer = false;
  }
}

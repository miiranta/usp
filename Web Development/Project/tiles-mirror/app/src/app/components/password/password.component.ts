import { Component, Input, Output, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { PlayerService } from '../../services/player.service';
import { LoadingService } from '../../services/loading.service';
import { ApiPlayerService } from '../../services/api-player.service';
import { PopupService } from '../../services/popup.service';

@Component({
  selector: 'app-password',
  imports: [FormsModule, CommonModule],
  templateUrl: './password.component.html',
  styleUrl: './password.component.scss',
})
export class PasswordComponent {
  @Input() playerName: string = '';
  @Input() isNewPlayer: boolean = false;
  @Output() backToChooseName = new EventEmitter<void>();

  password: string = '';
  confirmPassword: string = '';
  isLoading: boolean = false;
  constructor(
    private router: Router,
    private playerService: PlayerService,
    private loadingService: LoadingService,
    private apiPlayerService: ApiPlayerService,
    private popupService: PopupService,
  ) {}
  async handlePasswordSubmit() {
    if (!this.canSubmit()) {
      return;
    }

    this.isLoading = true;

    try {
      if (this.isNewPlayer) {
        await this.createPlayerAccount();
      } else {
        await this.loginPlayer();
      }
    } catch (error) {
      console.error('Erro:', error);
      this.popupService.error('Erro', 'Erro de conexão. Tente novamente.');
      this.isLoading = false;
    }
  }
  private async createPlayerAccount() {
    this.loadingService.show('Criando conta...');

    const response = await this.apiPlayerService.createPlayerWithPassword(
      this.playerName,
      this.password,
    );
    const data = await response.json();

    if (response.ok && data.success) {
      this.playerService.setPlayerName(data.playerName);
      this.playerService.setJwtToken(data.token);

      this.loadingService.setMessage('Entrando no jogo...');
      setTimeout(() => {
        this.router.navigate(['/tiles']);
      }, 1000);
    } else {
      this.loadingService.hide();
      this.isLoading = false;
      this.popupService.error(
        'Erro ao criar conta',
        data.error || 'Falha ao criar conta',
      );
    }
  }
  private async loginPlayer() {
    this.loadingService.show('Verificando senha...');

    const response = await this.apiPlayerService.authenticatePlayer(
      this.playerName,
      this.password,
    );
    const data = await response.json();

    if (response.ok && data.success) {
      this.playerService.setPlayerName(data.playerName);
      this.playerService.setJwtToken(data.token);

      this.loadingService.setMessage('Entrando no jogo...');
      setTimeout(() => {
        this.router.navigate(['/tiles']);
      }, 1000);
    } else {
      this.loadingService.hide();
      this.isLoading = false;
      this.popupService.error(
        'Erro de autenticação',
        data.error || 'Senha incorreta',
      );
    }
  }
  goBack() {
    this.backToChooseName.emit();
  }

  canSubmit(): boolean {
    if (this.isLoading) return false;

    if (this.isNewPlayer) {
      return (
        this.password.length >= 4 &&
        this.confirmPassword.length >= 4 &&
        this.password === this.confirmPassword
      );
    } else {
      return this.password.length >= 4;
    }
  }

  get passwordsMatch(): boolean {
    if (!this.isNewPlayer) return true;
    return this.password === this.confirmPassword;
  }

  get showPasswordMismatch(): boolean {
    return (
      this.isNewPlayer &&
      this.confirmPassword.length > 0 &&
      !this.passwordsMatch
    );
  }
}

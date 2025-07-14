import {
  Component,
  OnInit,
  inject,
  ViewChild,
  ElementRef,
  AfterViewInit,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { Location } from '@angular/common';
import { filter } from 'rxjs/operators';
import { EditService } from '../../services/edit.service';

declare global {
  interface Window {
    electronAPI: {
      minimizeWindow: () => Promise<void>;
      maximizeWindow: () => Promise<void>;
      closeWindow: () => Promise<void>;
      isMaximized: () => Promise<boolean>;
      onWindowMaximized: (callback: (isMaximized: boolean) => void) => void;
    };
  }
}

interface Breadcrumb {
  name: string;
  fullName: string;
  route: string;
}

@Component({
  selector: 'app-titlebar',
  imports: [CommonModule],
  templateUrl: './titlebar.component.html',
  styleUrl: './titlebar.component.scss',
})
export class TitlebarComponent implements OnInit, AfterViewInit {
  @ViewChild('rippleContainer', { static: true }) rippleContainer!: ElementRef;

  private router = inject(Router);
  private location = inject(Location);
  private editService = inject(EditService);

  isMaximized = false;
  currentLocation = '';
  canGoBack = false;
  isAtHome = true;
  breadcrumbs: Breadcrumb[] = [];

  private previousCanGoBack = false;
  private previousIsAtHome = true;

  ngOnInit() {
    this.checkMaximizedState();

    if (window.electronAPI) {
      window.electronAPI.onWindowMaximized((isMaximized: boolean) => {
        this.isMaximized = isMaximized;
      });
    }

    this.updateLocationInfo();
    this.router.events
      .pipe(filter((event) => event instanceof NavigationEnd))
      .subscribe(() => {
        this.updateLocationInfo();
      });
  }

  ngAfterViewInit() {
    this.previousCanGoBack = this.canGoBack;
    this.previousIsAtHome = this.isAtHome;
  }

  private updateLocationInfo() {
    const path = this.router.url;
    const newIsAtHome = path === '/';
    const newCanGoBack = window.history.length > 1 && !newIsAtHome;

    if (!this.previousCanGoBack && newCanGoBack) {
      this.triggerRippleAnimation('back');
    }

    if (this.previousIsAtHome && !newIsAtHome) {
      this.triggerRippleAnimation('home');
    }

    this.isAtHome = newIsAtHome;
    this.canGoBack = newCanGoBack;
    this.previousIsAtHome = newIsAtHome;
    this.previousCanGoBack = newCanGoBack;

    this.breadcrumbs = this.createBreadcrumbs(path);

    if (path === '/') {
      this.currentLocation = 'Home';
    } else if (path.startsWith('/viewPhoto/')) {
      this.currentLocation = 'Photo View';
    } else if (path.startsWith('/viewCollection/')) {
      this.currentLocation = 'Collection View';
    } else if (path === '/editPhoto') {
      this.currentLocation = 'Photo Editor';
    } else {
      this.currentLocation = 'PlantGrowthAnalyzer';
    }
  }
  private createBreadcrumbs(path: string): Breadcrumb[] {
    const breadcrumbs: Breadcrumb[] = [
      { name: 'Home', fullName: 'Home', route: '/' },
    ];

    if (path === '/') {
      return breadcrumbs;
    }

    if (path.startsWith('/viewPhoto/')) {
      breadcrumbs.push({
        name: 'View Photo',
        fullName: 'View Photo',
        route: path,
      });
    } else if (path.startsWith('/viewCollection/')) {
      breadcrumbs.push({
        name: 'View Collection',
        fullName: 'View Collection',
        route: path,
      });
    } else if (path.startsWith('/editPhoto')) {
      const previousRoute = this.editService.getPreviousRoute();

      if (previousRoute && previousRoute.startsWith('/viewPhoto/')) {
        breadcrumbs.push({
          name: 'View Photo',
          fullName: 'View Photo',
          route: previousRoute,
        });
      } else if (
        previousRoute &&
        previousRoute.startsWith('/viewCollection/')
      ) {
        breadcrumbs.push({
          name: 'View Collection',
          fullName: 'View Collection',
          route: previousRoute,
        });
      }

      breadcrumbs.push({
        name: 'Photo Editor',
        fullName: 'Photo Editor',
        route: path,
      });
    }

    return breadcrumbs;
  }

  async checkMaximizedState() {
    if (window.electronAPI) {
      this.isMaximized = await window.electronAPI.isMaximized();
    }
  }
  navigateToHome() {
    this.router.navigate(['/']);
  }

  navigateToHomeWithRipple() {
    this.router.navigate(['/']);
  }

  goBack() {
    this.location.back();
  }

  async minimizeWindow() {
    if (window.electronAPI) {
      await window.electronAPI.minimizeWindow();
    }
  }
  async maximizeWindow() {
    if (window.electronAPI) {
      await window.electronAPI.maximizeWindow();
    }
  }
  async closeWindow() {
    if (window.electronAPI) {
      await window.electronAPI.closeWindow();
    }
  }
  private triggerRippleAnimation(type: 'back' | 'home') {
    if (!this.rippleContainer) {
      return;
    }

    const titlebarHeight = 48;
    const buttonY = titlebarHeight / 2;
    const backButtonX = 50;
    const homeButtonX = 100;

    const centerX = type === 'back' ? backButtonX : homeButtonX;
    const centerY = buttonY;

    const ripple = document.createElement('div');

    ripple.style.position = 'fixed';
    ripple.style.left = centerX + 'px';
    ripple.style.top = centerY + 'px';
    ripple.style.width = '20px';
    ripple.style.height = '20px';
    ripple.style.marginLeft = '-10px';
    ripple.style.marginTop = '-10px';
    ripple.style.borderRadius = '50%';
    ripple.style.zIndex = '99999';
    ripple.style.pointerEvents = 'none';

    if (type === 'back') {
      ripple.style.background =
        'radial-gradient(circle, rgba(34, 197, 94, 0.5) 0%, rgba(34, 197, 94, 0.3) 20%, rgba(34, 197, 94, 0.15) 40%, rgba(34, 197, 94, 0.05) 60%, transparent 80%)';
      ripple.style.border = '1px solid rgba(34, 197, 94, 0.3)';
    } else {
      ripple.style.background =
        'radial-gradient(circle, rgba(34, 197, 94, 0.5) 0%, rgba(34, 197, 94, 0.3) 20%, rgba(34, 197, 94, 0.15) 40%, rgba(34, 197, 94, 0.05) 60%, transparent 80%)';
      ripple.style.border = '1px solid rgba(34, 197, 94, 0.3)';
    }

    ripple.style.transform = 'scale(0)';
    ripple.style.transition =
      'transform 1.8s cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 1.8s ease-out';
    ripple.style.opacity = '1';

    document.body.appendChild(ripple);

    ripple.offsetHeight;

    setTimeout(() => {
      ripple.style.transform = 'scale(18)';
      ripple.style.opacity = '0';
    }, 50);

    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple);
      }
    }, 1900);
  }

  private triggerRippleAnimationFromClick(type: 'back' | 'home') {
    const titlebarHeight = 48;
    const buttonY = titlebarHeight / 2;
    const backButtonX = 50;
    const homeButtonX = 100;

    const centerX = type === 'back' ? backButtonX : homeButtonX;
    const centerY = buttonY;

    const ripple = document.createElement('div');

    ripple.style.position = 'fixed';
    ripple.style.left = centerX + 'px';
    ripple.style.top = centerY + 'px';
    ripple.style.width = '24px';
    ripple.style.height = '24px';
    ripple.style.marginLeft = '-12px';
    ripple.style.marginTop = '-12px';
    ripple.style.borderRadius = '50%';
    ripple.style.zIndex = '99999';
    ripple.style.pointerEvents = 'none';

    if (type === 'back') {
      ripple.style.background =
        'radial-gradient(circle, rgba(34, 197, 94, 0.7) 0%, rgba(34, 197, 94, 0.4) 25%, rgba(34, 197, 94, 0.2) 50%, rgba(34, 197, 94, 0.1) 75%, transparent 90%)';
      ripple.style.border = '2px solid rgba(34, 197, 94, 0.4)';
    } else {
      ripple.style.background =
        'radial-gradient(circle, rgba(59, 130, 246, 0.7) 0%, rgba(59, 130, 246, 0.4) 25%, rgba(59, 130, 246, 0.2) 50%, rgba(59, 130, 246, 0.1) 75%, transparent 90%)';
      ripple.style.border = '2px solid rgba(59, 130, 246, 0.4)';
    }

    ripple.style.transform = 'scale(0)';
    ripple.style.transition =
      'transform 1.4s cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 1.4s ease-out';
    ripple.style.opacity = '1';

    document.body.appendChild(ripple);

    ripple.offsetHeight;

    setTimeout(() => {
      ripple.style.transform = 'scale(16)';
      ripple.style.opacity = '0';
    }, 30);

    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple);
      }
    }, 1500);
  }

  navigateToBreadcrumb(breadcrumb: Breadcrumb) {
    if (breadcrumb.route !== this.router.url) {
      if (breadcrumb.route === '/') {
        this.navigateToHome();
      } else {
        this.router.navigate([breadcrumb.route]).catch((err) => {
          this.navigateToHome();
        });
      }
    }
  }
}

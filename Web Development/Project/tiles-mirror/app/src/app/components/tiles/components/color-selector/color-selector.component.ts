import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-color-selector',
  imports: [CommonModule],
  templateUrl: './color-selector.component.html',
  styleUrl: './color-selector.component.scss',
})
export class ColorSelectorComponent {
  @Input() colors: string[] = [];
  @Input() selectedColor: string = '';
  @Output() colorSelected = new EventEmitter<string>();

  selectColor(color: string) {
    this.colorSelected.emit(color);
  }
}

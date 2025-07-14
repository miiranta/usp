import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EditPhotoPageComponent } from './edit-photo-page.component';

describe('EditPhotoPageComponent', () => {
  let component: EditPhotoPageComponent;
  let fixture: ComponentFixture<EditPhotoPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EditPhotoPageComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(EditPhotoPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

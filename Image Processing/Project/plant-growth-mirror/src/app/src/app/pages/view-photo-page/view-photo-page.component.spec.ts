import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ViewPhotoPageComponent } from './view-photo-page.component';

describe('ViewPhotoPageComponent', () => {
  let component: ViewPhotoPageComponent;
  let fixture: ComponentFixture<ViewPhotoPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewPhotoPageComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ViewPhotoPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

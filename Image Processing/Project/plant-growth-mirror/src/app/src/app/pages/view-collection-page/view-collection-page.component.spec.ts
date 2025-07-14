import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ViewCollectionPageComponent } from './view-collection-page.component';

describe('ViewCollectionPageComponent', () => {
  let component: ViewCollectionPageComponent;
  let fixture: ComponentFixture<ViewCollectionPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewCollectionPageComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ViewCollectionPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

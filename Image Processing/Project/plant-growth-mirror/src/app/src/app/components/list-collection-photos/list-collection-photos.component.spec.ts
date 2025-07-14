import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ListCollectionPhotosComponent } from './list-collection-photos.component';

describe('ListCollectionPhotosComponent', () => {
  let component: ListCollectionPhotosComponent;
  let fixture: ComponentFixture<ListCollectionPhotosComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ListCollectionPhotosComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ListCollectionPhotosComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

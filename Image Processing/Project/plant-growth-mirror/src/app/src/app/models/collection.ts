export interface Collection {
  id: string;
  name: string;
  photoIds?: string[];

  createdAt: Date;
  updatedAt?: Date;
}

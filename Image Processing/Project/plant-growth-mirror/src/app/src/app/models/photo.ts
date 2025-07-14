export interface Photo {
  id: string;
  collectionId?: string | null;

  dataBase64: string;
  dataProcessedBase64?: string;

  params?: {
    granularity?: number;
    threshold?: number;
  };

  result?: {
    height?: number;
    width?: number;
    area?: number;
  };

  createdAt?: Date;
  updatedAt?: Date;
  timestamp?: string;
}

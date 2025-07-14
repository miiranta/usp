import { Injectable } from '@angular/core';
import { Collection } from '../models/collection';
import { Photo } from '../models/photo';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private readonly apiUrl = 'http://localhost:3999/api';

  constructor() {}

  private normalizePhoto(raw: any): Photo {
    const incoming =
      raw.params && typeof raw.params === 'object' ? raw.params : {};

    return {
      id: raw.id,
      collectionId: raw.collectionId ?? null,
      dataBase64: raw.dataBase64,
      dataProcessedBase64: raw.dataProcessedBase64 ?? undefined,
      params: {
        granularity:
          typeof incoming.granularity === 'number' ? incoming.granularity : 1,
        threshold:
          typeof incoming.threshold === 'number' ? incoming.threshold : 128,
      },
      result: raw.result ?? undefined,
      createdAt: raw.createdAt ? new Date(raw.createdAt) : undefined,
      updatedAt: raw.updatedAt ? new Date(raw.updatedAt) : undefined,
    };
  }

  private normalizePhotoArray(rawArr: any[]): Photo[] {
    return rawArr.map((r) => this.normalizePhoto(r));
  }

  getCollections(): Promise<Collection[]> {
    return fetch(`${this.apiUrl}/collections`).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      return (await res.json()) as Collection[];
    });
  }

  updateCollection(collection: Collection): Promise<Collection> {
    return fetch(`${this.apiUrl}/collections/${collection.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collection),
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      return (await res.json()) as Collection;
    });
  }

  createCollection(name: string): Promise<Collection> {
    if (!name || name.trim() === '') {
      return Promise.reject(new Error('Collection name cannot be empty'));
    }
    return fetch(`${this.apiUrl}/collections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      return (await res.json()) as Collection;
    });
  }

  deleteCollection(id: string): Promise<void> {
    if (!id) {
      return Promise.reject(new Error('Collection ID is required'));
    }
    return fetch(`${this.apiUrl}/collections/${id}`, {
      method: 'DELETE',
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
    });
  }

  createOrEditPhoto(photo: Photo): Promise<Photo> {
    if (!photo || !photo.dataBase64) {
      return Promise.reject(new Error('Photo data is required'));
    }
    return fetch(`${this.apiUrl}/photos`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(photo),
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      const raw = await res.json();
      return this.normalizePhoto(raw);
    });
  }

  createOrEditPhotoNoProc(photo: Photo): Promise<Photo> {
    if (!photo || !photo.dataBase64) {
      return Promise.reject(new Error('Photo data is required'));
    }
    return fetch(`${this.apiUrl}/photos/noProc`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(photo),
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      const raw = await res.json();
      return this.normalizePhoto(raw);
    });
  }

  getPhoto(id: string): Promise<Photo | null> {
    if (!id) {
      return Promise.resolve(null);
    }
    return fetch(`${this.apiUrl}/photos/${id}`).then(async (res) => {
      if (res.status === 404) {
        return null;
      }
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      const raw = await res.json();
      return this.normalizePhoto(raw);
    });
  }

  getPhotos(): Promise<Photo[]> {
    return fetch(`${this.apiUrl}/photos`).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      const rawArr = (await res.json()) as any[];
      return this.normalizePhotoArray(rawArr);
    });
  }

  deletePhoto(id: string): Promise<void> {
    if (!id) {
      return Promise.reject(new Error('Photo ID is required'));
    }
    return fetch(`${this.apiUrl}/photos/${id}`, {
      method: 'DELETE',
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
    });
  }

  updatePhotoCollectionId(
    photoId: string,
    collectionId: string | null,
  ): Promise<Photo> {
    if (!photoId) {
      return Promise.reject(new Error('Photo ID is required'));
    }
    return fetch(`${this.apiUrl}/photos/${photoId}/collection`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collectionId }),
    }).then(async (res) => {
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Error ${res.status}: ${errText}`);
      }
      const raw = await res.json();
      return this.normalizePhoto(raw);
    });
  }
}

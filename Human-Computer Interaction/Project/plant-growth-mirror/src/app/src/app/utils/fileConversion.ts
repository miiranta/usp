export interface ImageMetadata {
  width: number;
  height: number;
  mimeType: string;
  exif: Record<string, any>;
}

export class FileConversion {
  static fileToBase64(file: File): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result);
        } else {
          reject('FileReader result is not a string');
        }
      };

      reader.onerror = (error) => reject(error);
    });
  }
}

// utils/imageValidation.js
/**
 * Validates if a base64 string represents a valid image format
 * @param {string} dataBase64 - Base64 image data (with or without data URL prefix)
 * @returns {Object} Validation result with isValid boolean and details
 */
export function validateImageFormat(dataBase64) {
  try {
    // Check if it's a data URL
    if (!dataBase64 || typeof dataBase64 !== "string") {
      return {
        isValid: false,
        error: "Invalid data: not a string",
        mimeType: null,
      };
    }

    let mimeType = null;
    let base64Data = dataBase64;

    // Extract MIME type from data URL
    if (dataBase64.startsWith("data:")) {
      const match = dataBase64.match(/^data:([^;]+);base64,(.+)$/);
      if (!match) {
        return {
          isValid: false,
          error: "Invalid data URL format",
          mimeType: null,
        };
      }
      mimeType = match[1];
      base64Data = match[2];
    }

    // Validate base64 string
    if (!isValidBase64(base64Data)) {
      return {
        isValid: false,
        error: "Invalid base64 encoding",
        mimeType,
      };
    }

    // Accepted image MIME types
    const acceptedMimeTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/gif",
      "image/webp",
      "image/tiff",
      "image/tif",
      "image/bmp",
    ];

    // If we have a MIME type, validate it
    if (mimeType) {
      if (!acceptedMimeTypes.includes(mimeType.toLowerCase())) {
        return {
          isValid: false,
          error: `Unsupported image format: ${mimeType}. Accepted formats: ${acceptedMimeTypes.join(", ")}`,
          mimeType,
        };
      }
    } else {
      // Try to detect format from magic bytes
      const detectedType = detectImageTypeFromBytes(base64Data);
      if (!detectedType) {
        return {
          isValid: false,
          error:
            "Could not detect image format. Accepted formats: " +
            acceptedMimeTypes.join(", "),
          mimeType: null,
        };
      }
      mimeType = detectedType;
    }

    return {
      isValid: true,
      error: null,
      mimeType,
    };
  } catch (error) {
    return {
      isValid: false,
      error: `Validation error: ${error.message}`,
      mimeType: null,
    };
  }
}

/**
 * Validates if a string is valid base64
 * @param {string} str - String to validate
 * @returns {boolean} True if valid base64
 */
function isValidBase64(str) {
  try {
    // Check if string contains only valid base64 characters
    const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Regex.test(str)) {
      return false;
    }

    // Try to decode
    Buffer.from(str, "base64");
    return true;
  } catch {
    return false;
  }
}

/**
 * Detects image type from the first few bytes (magic bytes)
 * @param {string} base64Data - Base64 encoded image data
 * @returns {string|null} MIME type or null if not detected
 */
function detectImageTypeFromBytes(base64Data) {
  try {
    // Get first 12 bytes to check magic bytes
    const buffer = Buffer.from(base64Data.substring(0, 16), "base64");
    const bytes = Array.from(buffer);

    // JPEG: FF D8 FF
    if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
      return "image/jpeg";
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (
      bytes[0] === 0x89 &&
      bytes[1] === 0x50 &&
      bytes[2] === 0x4e &&
      bytes[3] === 0x47
    ) {
      return "image/png";
    }

    // GIF: 47 49 46 38 (GIF8)
    if (
      bytes[0] === 0x47 &&
      bytes[1] === 0x49 &&
      bytes[2] === 0x46 &&
      bytes[3] === 0x38
    ) {
      return "image/gif";
    }

    // WebP: 52 49 46 46 (RIFF) ... 57 45 42 50 (WEBP)
    if (
      bytes[0] === 0x52 &&
      bytes[1] === 0x49 &&
      bytes[2] === 0x46 &&
      bytes[3] === 0x46 &&
      bytes[8] === 0x57 &&
      bytes[9] === 0x45 &&
      bytes[10] === 0x42 &&
      bytes[11] === 0x50
    ) {
      return "image/webp";
    }

    // BMP: 42 4D (BM)
    if (bytes[0] === 0x42 && bytes[1] === 0x4d) {
      return "image/bmp";
    }

    // TIFF: 49 49 2A 00 (little endian) or 4D 4D 00 2A (big endian)
    if (
      (bytes[0] === 0x49 &&
        bytes[1] === 0x49 &&
        bytes[2] === 0x2a &&
        bytes[3] === 0x00) ||
      (bytes[0] === 0x4d &&
        bytes[1] === 0x4d &&
        bytes[2] === 0x00 &&
        bytes[3] === 0x2a)
    ) {
      return "image/tiff";
    }

    return null;
  } catch {
    return null;
  }
}

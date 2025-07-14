// router/router.js
import { Router } from "express";
import crypto from "crypto";
import database from "../database/database.js";
import processImage from "../image-processing/processImage.js";
import { validateImageFormat } from "../utils/imageValidation.js";

const { collections: collectionsDb, photos: photosDb } = database.db;

// Default values for any missing params fields
const DEFAULT_GRANULARITY = 20;
const DEFAULT_THRESHOLD = 0.05;

function generateId() {
  return crypto.randomBytes(8).toString("hex");
}

// Ensure that any missing params get default values before returning
function fillInParams(photo) {
  const raw = photo.params || {};
  return {
    ...photo,
    params: {
      granularity:
        typeof raw.granularity === "number"
          ? raw.granularity
          : DEFAULT_GRANULARITY,
      threshold:
        typeof raw.threshold === "number" ? raw.threshold : DEFAULT_THRESHOLD,
    },
  };
}

const setupEndpoints = (app) => {
  const router = Router();
  app.use("/api", router);

  //
  // Collections ---
  //

  /**
   * GET /api/collections
   */
  router.get("/collections", (req, res) => {
    collectionsDb.find({}, (err, docs) => {
      if (err) {
        console.error("Error fetching collections:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      // Return as an array of plain objects
      return res.json(docs);
    });
  });

  /**
   * POST /api/collections
   */
  router.post("/collections", (req, res) => {
    const { name } = req.body;
    if (!name || typeof name !== "string" || name.trim() === "") {
      return res.status(400).json({ error: "Collection name cannot be empty" });
    }

    const newCollection = {
      id: generateId(),
      name: name.trim(),
      photoIds: [],
      createdAt: new Date().toISOString(),
      updatedAt: null,
    };

    collectionsDb.insert(newCollection, (err, inserted) => {
      if (err) {
        console.error("Error inserting collection:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      return res.status(201).json(inserted);
    });
  });

  /**
   * PUT /api/collections/:id
   */
  router.put("/collections/:id", (req, res) => {
    const { id } = req.params;
    const body = req.body;

    if (!body || body.id !== id) {
      return res
        .status(400)
        .json({ error: "Invalid collection data or ID mismatch" });
    }

    // Find existing collection
    collectionsDb.findOne({ id }, (err, existing) => {
      if (err) {
        console.error("Error finding collection:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      if (!existing) {
        return res.status(404).json({ error: "Collection not found" });
      }

      const updated = {
        ...existing,
        ...body,
        updatedAt: new Date().toISOString(),
      };

      collectionsDb.update(
        { id },
        updated,
        { returnUpdatedDocs: true },
        (err2, numAffected, affectedDoc) => {
          if (err2) {
            console.error("Error updating collection:", err2);
            return res.status(500).json({ error: "Internal server error" });
          }
          return res.json(affectedDoc);
        },
      );
    });
  });

  /**
   * DELETE /api/collections/:id
   */
  router.delete("/collections/:id", (req, res) => {
    const { id } = req.params;

    // First, remove the collection document
    collectionsDb.remove({ id }, {}, (err, numRemoved) => {
      if (err) {
        console.error("Error deleting collection:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      if (numRemoved === 0) {
        return res.status(404).json({ error: "Collection not found" });
      }

      // Next, for any photos that referenced this collectionId, set collectionId to null
      photosDb.update(
        { collectionId: id },
        { $set: { collectionId: null } },
        { multi: true },
        (err2) => {
          if (err2) {
            console.error("Error clearing collectionId on photos:", err2);
            // Even if this step fails, we've already removed the collection. So still return 204.
          }
          return res.status(204).send();
        },
      );
    });
  });

  //
  // Photos ---
  //

  /**
   * GET /api/photos
   * Return all photos.
   */
  router.get("/photos", (req, res) => {
    photosDb.find({}, (err, docs) => {
      if (err) {
        console.error("Error fetching photos:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      // Apply fillInParams to each
      const all = docs.map((p) => fillInParams(p));
      return res.json(all);
    });
  });

  /**
   * GET /api/photos/:id
   */
  router.get("/photos/:id", (req, res) => {
    const { id } = req.params;
    photosDb.findOne({ id }, (err, photo) => {
      if (err) {
        console.error("Error fetching photo:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      if (!photo) {
        return res.status(404).json({ error: "Photo not found" });
      }
      return res.json(fillInParams(photo));
    });
  });

  /**
   * POST /api/photos
   * Create or update (with “processing”).
   */
  router.post("/photos", async (req, res) => {
    const body = req.body;
    if (!body || !body.dataBase64) {
      return res.status(400).json({ error: "Photo data is required" });
    }

    // Validate image format
    const validation = validateImageFormat(body.dataBase64);
    if (!validation.isValid) {
      return res.status(400).json({
        error: `Invalid image format: ${validation.error}`,
        details:
          "Only image files are accepted (JPG, PNG, GIF, WebP, TIFF, BMP)",
      });
    }

    const incomingId = body.id || null;

    if (!incomingId) {
      // CREATE new photo
      const newPhoto = {
        id: generateId(),
        collectionId: body.collectionId ?? null,
        dataBase64: body.dataBase64,
        dataProcessedBase64: null,
        params: {
          granularity:
            typeof body.params?.granularity === "number"
              ? body.params.granularity
              : DEFAULT_GRANULARITY,
          threshold:
            typeof body.params?.threshold === "number"
              ? body.params.threshold
              : DEFAULT_THRESHOLD,
        },
        result: null,
        createdAt: body.timestamp ? body.timestamp : new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      photosDb.insert(newPhoto, (err, inserted) => {
        if (err) {
          console.error("Error inserting photo:", err);
          return res.status(500).json({ error: "Internal server error" });
        }
        return res.status(201).json(fillInParams(inserted));
      });
    } else {
      // UPDATE existing photo (with processing)
      photosDb.findOne({ id: incomingId }, async (err, old) => {
        if (err) {
          console.error("Error finding photo:", err);
          return res.status(500).json({ error: "Internal server error" });
        }
        if (!old) {
          // If it doesn't exist, treat as create
          const newPhoto = {
            id: generateId(),
            collectionId: body.collectionId ?? null,
            dataBase64: body.dataBase64,
            dataProcessedBase64: null,
            params: {
              granularity:
                typeof body.params?.granularity === "number"
                  ? body.params.granularity
                  : DEFAULT_GRANULARITY,
              threshold:
                typeof body.params?.threshold === "number"
                  ? body.params.threshold
                  : DEFAULT_THRESHOLD,
            },
            result: null,
            createdAt: body.timestamp
              ? body.timestamp
              : new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };

          photosDb.insert(newPhoto, (err2, inserted) => {
            if (err2) {
              console.error("Error inserting photo:", err2);
              return res.status(500).json({ error: "Internal server error" });
            }
            return res.status(201).json(fillInParams(inserted));
          });
        } else {
          // Exists: update with processing

          const processedData = processImage(
            old.id,
            body.dataBase64,
            body.params?.granularity ??
              old.params?.granularity ??
              DEFAULT_GRANULARITY,
            body.params?.threshold ??
              old.params?.threshold ??
              DEFAULT_THRESHOLD,
          );

          processedData.then((processedData) => {
            const updatedPhoto = {
              ...old,
              dataBase64: body.dataBase64,
              collectionId:
                body.collectionId !== undefined
                  ? body.collectionId
                  : old.collectionId,
              params: {
                granularity:
                  typeof body.params?.granularity === "number"
                    ? body.params.granularity
                    : (old.params?.granularity ?? DEFAULT_GRANULARITY),
                threshold:
                  typeof body.params?.threshold === "number"
                    ? body.params.threshold
                    : (old.params?.threshold ?? DEFAULT_THRESHOLD),
              },
              updatedAt: new Date().toISOString(),
              createdAt: body.createdAt || old.createdAt,
              result: {
                width: processedData.result.width || 0,
                height: processedData.result.height || 0,
                area: processedData.result.area || 0,
              },
              dataProcessedBase64: processedData.dataProcessedBase64 || null,
            };

            photosDb.update(
              { id: incomingId },
              updatedPhoto,
              { returnUpdatedDocs: true },
              (err2, _, affectedDoc) => {
                if (err2) {
                  console.error("Error updating photo:", err2);
                  return res
                    .status(500)
                    .json({ error: "Internal server error" });
                }
                return res.json(fillInParams(affectedDoc));
              },
            );
          });
        }
      });
    }
  });

  /**
   * POST /api/photos/noProc
   * Create or update without processing.
   */ router.post("/photos/noProc", async (req, res) => {
    const body = req.body;
    if (!body || !body.dataBase64) {
      return res.status(400).json({ error: "Photo data is required" });
    }

    // Validate image format
    const validation = validateImageFormat(body.dataBase64);
    if (!validation.isValid) {
      return res.status(400).json({
        error: `Invalid image format: ${validation.error}`,
        details:
          "Only image files are accepted (JPG, PNG, GIF, WebP, TIFF, BMP)",
      });
    }

    const incomingId = body.id || null;

    if (!incomingId) {
      // CREATE new photo
      const newPhoto = {
        id: generateId(),
        collectionId: body.collectionId ?? null,
        dataBase64: body.dataBase64,
        dataProcessedBase64: null,
        params: {
          granularity:
            typeof body.params?.granularity === "number"
              ? body.params.granularity
              : DEFAULT_GRANULARITY,
          threshold:
            typeof body.params?.threshold === "number"
              ? body.params.threshold
              : DEFAULT_THRESHOLD,
        },
        result: null,
        createdAt: body.timestamp ? body.timestamp : new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      photosDb.insert(newPhoto, (err, inserted) => {
        if (err) {
          console.error("Error inserting photo:", err);
          return res.status(500).json({ error: "Internal server error" });
        }
        return res.status(201).json(fillInParams(inserted));
      });
    } else {
      // UPDATE existing photo (no processing)
      photosDb.findOne({ id: incomingId }, async (err, old) => {
        if (err) {
          console.error("Error finding photo:", err);
          return res.status(500).json({ error: "Internal server error" });
        }
        if (!old) {
          // If it doesn't exist, treat as create
          const newPhoto = {
            id: generateId(),
            collectionId: body.collectionId ?? null,
            dataBase64: body.dataBase64,
            dataProcessedBase64: null,
            params: {
              granularity:
                typeof body.params?.granularity === "number"
                  ? body.params.granularity
                  : DEFAULT_GRANULARITY,
              threshold:
                typeof body.params?.threshold === "number"
                  ? body.params.threshold
                  : DEFAULT_THRESHOLD,
            },
            result: null,
            createdAt: body.timestamp
              ? body.timestamp
              : new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };

          photosDb.insert(newPhoto, (err2, inserted) => {
            if (err2) {
              console.error("Error inserting photo:", err2);
              return res.status(500).json({ error: "Internal server error" });
            }
            return res.status(201).json(fillInParams(inserted));
          });
        } else {
          // Exists: update without processing
          const updatedPhoto = {
            ...body,
          };

          photosDb.update(
            { id: incomingId },
            updatedPhoto,
            { returnUpdatedDocs: true },
            (err2, _, affectedDoc) => {
              if (err2) {
                console.error("Error updating photo:", err2);
                return res.status(500).json({ error: "Internal server error" });
              }
              return res.json(fillInParams(affectedDoc));
            },
          );
        }
      });
    }
  });

  /**
   * DELETE /api/photos/:id
   */
  router.delete("/photos/:id", (req, res) => {
    const { id } = req.params;

    photosDb.remove({ id }, {}, (err, numRemoved) => {
      if (err) {
        console.error("Error deleting photo:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      if (numRemoved === 0) {
        return res.status(404).json({ error: "Photo not found" });
      }

      // Also remove this photo ID from any collection’s photoIds array
      collectionsDb.update(
        { photoIds: id },
        { $pull: { photoIds: id } },
        { multi: true },
        (err2) => {
          if (err2) {
            console.error("Error removing photoId from collections:", err2);
            // Photo is already removed; respond with 204 regardless
          }
          return res.status(204).send();
        },
      );
    });
  });

  /**
   * PATCH /api/photos/:photoId/collection
   */
  router.patch("/photos/:photoId/collection", (req, res) => {
    const { photoId } = req.params;
    const { collectionId } = req.body;

    photosDb.findOne({ id: photoId }, (err, photo) => {
      if (err) {
        console.error("Error finding photo:", err);
        return res.status(500).json({ error: "Internal server error" });
      }
      if (!photo) {
        return res.status(404).json({ error: "Photo not found" });
      }

      if (collectionId !== null) {
        // Check that the target collection exists
        collectionsDb.findOne({ id: collectionId }, (err2, collection) => {
          if (err2) {
            console.error("Error finding collection:", err2);
            return res.status(500).json({ error: "Internal server error" });
          }
          if (!collection) {
            return res.status(400).json({ error: "Collection not found" });
          }

          // Add photoId to that collection’s photoIds array if not already present
          if (!Array.isArray(collection.photoIds)) {
            collection.photoIds = [];
          }
          if (!collection.photoIds.includes(photoId)) {
            collection.photoIds.push(photoId);
          }

          collectionsDb.update(
            { id: collectionId },
            { $set: { photoIds: collection.photoIds } },
            {},
            (err3) => {
              if (err3) {
                console.error("Error updating collection.photoIds:", err3);
                // Even if this fails, we’ll still attempt to update the photo’s collectionId
              }
              // Finally update the photo’s collectionId field
              const updatedPhoto = {
                ...photo,
                collectionId,
                updatedAt: new Date().toISOString(),
              };
              photosDb.update(
                { id: photoId },
                updatedPhoto,
                { returnUpdatedDocs: true },
                (err4, _, affectedDoc) => {
                  if (err4) {
                    console.error("Error updating photo:", err4);
                    return res
                      .status(500)
                      .json({ error: "Internal server error" });
                  }
                  return res.json(fillInParams(affectedDoc));
                },
              );
            },
          );
        });
      } else {
        // collectionId is null: remove from any collection that has this photoId
        collectionsDb.update(
          { photoIds: photoId },
          { $pull: { photoIds: photoId } },
          { multi: true },
          (err2) => {
            if (err2) {
              console.error("Error removing photoId from collections:", err2);
              // Continue to update the photo anyway
            }
            const updatedPhoto = {
              ...photo,
              collectionId: null,
              updatedAt: new Date().toISOString(),
            };
            photosDb.update(
              { id: photoId },
              updatedPhoto,
              { returnUpdatedDocs: true },
              (err3, _, affectedDoc) => {
                if (err3) {
                  console.error("Error updating photo:", err3);
                  return res
                    .status(500)
                    .json({ error: "Internal server error" });
                }
                return res.json(fillInParams(affectedDoc));
              },
            );
          },
        );
      }
    });
  });
};

export default setupEndpoints;

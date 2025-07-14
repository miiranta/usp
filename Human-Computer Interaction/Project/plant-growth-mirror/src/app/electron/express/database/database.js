import Datastore from "@seald-io/nedb";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let collectionsDb = null;
let photosDb = null;

function loadDatabase() {
  if (!collectionsDb || !photosDb) {
    const dataDir = path.resolve(__dirname, "data");
    const collectionsPath = path.resolve(__dirname, "data/collections.db");
    const photosPath = path.resolve(__dirname, "data/photos.db");

    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    if (!fs.existsSync(collectionsPath)) {
      fs.writeFileSync(collectionsPath, "");
    }
    if (!fs.existsSync(photosPath)) {
      fs.writeFileSync(photosPath, "");
    }

    // Now create the NeDB-promises datastores
    collectionsDb = new Datastore({
      filename: collectionsPath,
      autoload: true,
    });
    photosDb = new Datastore({
      filename: photosPath,
      autoload: true,
    });

    console.log("NeDB is up.");
  }

  return { collections: collectionsDb, photos: photosDb };
}

const db = loadDatabase();

export default {
  loadDatabase,
  db,
};

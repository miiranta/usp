// index.js
import express from "express";
import { createServer } from "http";
import setupEndpoints from "./router/router.js";
import database from "./database/database.js";
import { dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const angularAppPath = __dirname + "/../../dist/plant-growth/browser";

const PORT = 3999;
const expressApp = express();

expressApp.use(express.json({ limit: "500mb" }));
expressApp.use(express.urlencoded({ limit: "500mb", extended: true }));

expressApp.use(express.static(angularAppPath));

expressApp.get("/", (req, res) => {
  res.sendFile("index.html", { root: angularAppPath }, (err) => {
    if (err) {
      console.error("Error sending index.html:", err);
      res.status(500).send("Internal Server Error");
    }
  });
});

database.loadDatabase();
setupEndpoints(expressApp);

const server = createServer(expressApp);
server.listen(PORT, () => {
  console.log(`Express is up (${PORT}).`);
});

export default expressApp;

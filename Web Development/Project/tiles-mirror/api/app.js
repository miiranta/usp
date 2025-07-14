const path = require("path");
const { log } = require("./src/utils/colorLogging");

const dotenvPath = path.join(__dirname, "environments", ".env");
require("dotenv").config({ path: dotenvPath });

if (process.env.SSL == "true" && process.env.EMAIL && process.env.DOMAIN) {
  require("./src/main-ssl");
} else {
  require("./src/main");
}

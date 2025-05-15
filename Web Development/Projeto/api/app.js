const express     = require('express');
const http        = require('http');
const { Server }  = require("socket.io");
const path        = require('path');
const cors        = require('cors');

const { Tile, TileMap, COLORS } = require('./game');

const PORT = 3000;

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: '*'
  }
});

app.use(cors({
  origin: '*'
}));

const angular_folder = path.join(__dirname, '../app/dist/tiles/browser');

// Public folder
app.use(express.static(angular_folder));

// Game objects
var map = new TileMap();

// Envia o app Angular ao cliente
app.get('/', (req, res) => {
  res.sendFile(path.join(angular_folder, '/index.html'));
});

// WebSockets
io.on('connection', (socket) => {

  // Player postions
  socket.on('playerPosition', (data) => {
    io.emit('playerPosition', data);
  });

  // Player colors
  socket.on('tilePlaced', (data) => {
    io.emit('tilePlaced', data);
  });

  // Disconnect
  socket.on('disconnect', () => {});

});

// Endpoints REST
app.get('/map/:x/:y/:render', (req, res) => { // GET tiles
  if (!req.params.x || !req.params.y || !req.params.render) {
    return res.status(400).send('Missing parameters');
  }

  const x = parseInt(req.params.x);
  const y = parseInt(req.params.y);
  const render = req.params.render;

  if(render > 100) res.status(400).send('Render value too high (>100)');
  if(render < 0) res.status(400).send('Render value too low (<0)');
  if(isNaN(x) || isNaN(y)) res.status(400).send('Invalid coordinates');

  var tiles = [];
  for(let i = -Math.floor(render / 2); i <= Math.floor(render / 2); i++) {
    for(let j = -Math.floor(render / 2); j <= Math.floor(render / 2); j++) {
      const tile = map.getTile(x + i, y + j);
      if(tile) {
        tiles.push({
          x: tile.x,
          y: tile.y,
          type: tile.type
        });
      }
    }
  }

  res.json(tiles);
});

app.put('/map/:x/:y/:type', (req, res) => { // PUT tiles
  if (!req.params.x || !req.params.y || !req.params.type) {
    return res.status(400).send('Missing parameters');
  }

  const x = parseInt(req.params.x);
  const y = parseInt(req.params.y);
  const type = req.params.type;

  if(!COLORS.includes(type)) res.status(400).send('Invalid color');
  if(isNaN(x) || isNaN(y)) res.status(400).send('Invalid coordinates');

  map.placeTile(x, y, type);

  return res.sendStatus(200);
});

// Inicializa o servidor
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});


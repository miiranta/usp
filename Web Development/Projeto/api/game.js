const COLORS = [
  'white',
  'lightgray',
  'gray',
  'darkgray',
  'black',
  'red',
  'green',
  'blue',
  'yellow',
  'purple',
  'orange'
];

class Tile {
  type = 'white';
  x = 0;
  y = 0;
  
  constructor(x, y, type='notset') {
    this.x = x;
    this.y = y;

    if (type == 'notset') {
        // Set tile type based on coordinates - grid of white and light gray
        if ((x % 2 + 2) % 2 === (y % 2 + 2) % 2) {
        this.type = 'white';
        } else {
        this.type = 'lightgray';
        }
    } else {
        this.type = type;
    }

  }

}

class TileMap {
  constructor() {}

  map = new Map();

  getTile(x, y) {
    return this.createTile(x, y);
  }

  createTile(x, y) {
    const key = `${x},${y}`;
    if (this.map.has(key)) {
      const tile = this.map.get(key);
      if (tile) {
        return tile;
      }
      return null;
    }
    // Create a new tile and add it to the map
    const tile = new Tile(x, y);
    this.map.set(key, tile);

    return new Tile(x, y);
  }

  placeTile(x, y, type) {
    const key = `${x},${y}`;
    if (this.map.has(key)) {
      const tile = this.map.get(key);
      if (!tile) return;
      tile.type = type;
    } else {
      // Create a new tile and add it to the map
      const tile = new Tile(x, y);
      tile.type = type;
      this.map.set(key, tile);
    }
  }

}

module.exports = {
    TileMap,
    Tile,
    COLORS
};


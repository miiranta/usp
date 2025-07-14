import { Tile as ITile } from '../../interfaces/tile.interface';

export class Tile implements ITile {
  type = 'white';
  x = 0;
  y = 0;

  constructor(x: number, y: number, type: string = 'notset') {
    this.x = x;
    this.y = y;
    if (type == 'notset') {
      if (((x % 2) + 2) % 2 === ((y % 2) + 2) % 2) {
        this.type = 'white';
      } else {
        this.type = 'lightgray';
      }
    } else {
      this.type = type;
    }
  }
}

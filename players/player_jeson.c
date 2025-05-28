#include "../public/player.h"

#include <stdlib.h>

ShipPlacement jesonPlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT:
      return (ShipPlacement){0, 1, 0};
    case SHIP_SUBMARINE:
      return (ShipPlacement){3, 1, 1};
    case SHIP_DESTROYER:
      return (ShipPlacement){7, 2, 1};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){2, 5, 1};
    case SHIP_CARRIER:
      return (ShipPlacement){6, 6, 1};
  }
}
Coordinate jesonAction(Board board) {
  int x = -1, y = -1;
  do {
    x = rand() % 10, y = rand() % 10;
  } while (!checkShot(&board, (Coordinate){x, y}));
  return (Coordinate){x, y};
}
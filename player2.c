#include "public/player2.h"

#include <stdlib.h>

ShipPlacement player2PlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT:
      return (ShipPlacement){1, 1, 0};
    case SHIP_SUBMARINE:
      return (ShipPlacement){3, 1, 0};
    case SHIP_DESTROYER:
      return (ShipPlacement){7, 2, 0};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){1, 5, 0};
    case SHIP_CARRIER:
      return (ShipPlacement){5, 5, 0};
  }
}
Coordinate player2Action(Board board) {
  // return (Coordinate){-1, -1};
  int x, y;
  do {
    x = rand() % 10, y = rand() % 10;
  } while (!checkShot(&board, (Coordinate){x, y}));
  return (Coordinate){x, y};
}
#include "board_api.h"

// Python helper func

ShipPlacement playerPlaceShipPython(Board board, ShipID ship, PyObject *placeFnc);
Coordinate playerActionPython(Board board, PyObject *actionFnc);

// Players

ShipPlacement ethanPlaceShip(Board board, ShipID ship);
Coordinate ethanAction(Board board);

ShipPlacement jesonPlaceShip(Board board, ShipID ship);
Coordinate jesonAction(Board board);

ShipPlacement tomPlaceShip(Board board, ShipID ship);
Coordinate tomAction(Board board);

ShipPlacement vinPlaceShip(Board board, ShipID ship);
Coordinate vinAction(Board board);

ShipPlacement yashPlaceShip(Board board, ShipID ship);
Coordinate yashAction(Board board);
#include "public/player1.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef PYTHON_BOT
ShipPlacement player1PlaceShip(Board board, ShipID ship) {
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
Coordinate player1Action(Board board) {
  int x = -1, y = -1;
  do {
    x = rand() % 10, y = rand() % 10;
  } while (!checkShot(&board, (Coordinate){x, y}));
  return (Coordinate){x, y};
}
#else

ShipPlacement player1PlaceShip(Board board, ShipID ship) {
  PyObject *args = PyTuple_New(3);
  PyObject *pyBoard = PyList_New(BOARD_SIZE * BOARD_SIZE);
  PyObject *pyRemainingShips = PyList_New(SHIP_CARRIER + 1);
  PyObject *pyShip = PyLong_FromLong((long)ship);
  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    PyList_SET_ITEM(pyBoard, i, PyLong_FromLong((long)board.board[i]));
  }
  for (int i = 0; i < SHIP_CARRIER + 1; ++i) {
    PyList_SET_ITEM(pyRemainingShips, i, PyBool_FromLong((long)board.remaining_ships[i]));
  }
  PyTuple_SET_ITEM(args, 0, pyBoard), PyTuple_SET_ITEM(args, 1, pyRemainingShips), PyTuple_SET_ITEM(args, 2, pyShip);

  PyObject *placement = PyObject_CallObject(placeFnc, args);
  Py_DECREF(args), args = NULL;
  Py_DECREF(pyBoard), pyBoard = NULL;
  Py_DECREF(pyRemainingShips), pyRemainingShips = NULL;
  Py_DECREF(pyShip), pyShip = NULL;

  if (placement == NULL || !PyList_Check(placement) || PyList_GET_SIZE(placement) != 3) {
    if (placement != NULL)
      Py_DECREF(placement), placement = NULL;
    return (ShipPlacement){-1, -1, -1};
  }

  ShipPlacement result = (ShipPlacement){(int)PyLong_AsLong(PyList_GET_ITEM(placement, 0)),
                                         (int)PyLong_AsLong(PyList_GET_ITEM(placement, 1)),
                                         (int)PyLong_AsLong(PyList_GET_ITEM(placement, 2))};
  Py_DECREF(placement);
  return result;
}
Coordinate player1Action(Board board) {
  PyObject *args = PyTuple_New(3);
  PyObject *pyBoard = PyList_New(BOARD_SIZE * BOARD_SIZE);
  PyObject *pyRemainingShips = PyList_New(SHIP_CARRIER + 1);
  PyObject *pyHistory = PyList_New(board.history_head_ptr + 1);

  for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    PyList_SET_ITEM(pyBoard, i, PyLong_FromLong((long)board.board[i]));
  }
  for (int i = 0; i <= SHIP_CARRIER; ++i) {
    PyList_SET_ITEM(pyRemainingShips, i, PyBool_FromLong((long)board.remaining_ships[i]));
  }
  printf("%d\n", board.history_head_ptr);
  for (int i = 0; i <= board.history_head_ptr; ++i) {
    PyObject *action = PyTuple_New(3);
    PyTuple_SET_ITEM(action, 0, PyLong_FromLong((long)(board.history[i].pos.x)));
    PyTuple_SET_ITEM(action, 1, PyLong_FromLong((long)(board.history[i].pos.y)));
    PyTuple_SET_ITEM(action, 2, PyLong_FromLong((long)(board.history[i].hit)));
    PyList_SetItem(pyHistory, i, action);
    printf("%d| %d %d %d\n", i, board.history[i].pos.x, board.history[i].pos.y, board.history[i].hit);
  }
  printf("\n");

  PyTuple_SET_ITEM(args, 0, pyBoard), PyTuple_SET_ITEM(args, 1, pyRemainingShips), PyTuple_SET_ITEM(args, 2, pyHistory);

  PyObject *placement = PyObject_CallObject(actionFnc, args);
  Py_DECREF(args), args = NULL;
  Py_DECREF(pyBoard), pyBoard = NULL;
  Py_DECREF(pyRemainingShips), pyRemainingShips = NULL;

  if (placement == NULL || !PyList_Check(placement) || PyList_GET_SIZE(placement) != 2) {
    if (placement != NULL)
      Py_DECREF(placement), placement = NULL;
    return (Coordinate){-1, -1};
  }

  Coordinate result = (Coordinate){(int)PyLong_AsLong(PyList_GET_ITEM(placement, 0)),
                                   (int)PyLong_AsLong(PyList_GET_ITEM(placement, 1))};
  Py_DECREF(placement);
  return result;
}

#endif
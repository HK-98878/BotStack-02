#include "public/player.h"

#include <stdlib.h>

ShipPlacement playerPlaceShipPython(Board board, ShipID ship, PyObject *placeFnc) {
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
Coordinate playerActionPython(Board board, PyObject *actionFnc) {
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
  for (int i = 0; i <= board.history_head_ptr; ++i) {
    PyObject *action = PyTuple_New(3);
    PyTuple_SET_ITEM(action, 0, PyLong_FromLong((long)(board.history[i].pos.x)));
    PyTuple_SET_ITEM(action, 1, PyLong_FromLong((long)(board.history[i].pos.y)));
    PyTuple_SET_ITEM(action, 2, PyLong_FromLong((long)(board.history[i].hit)));
    PyList_SetItem(pyHistory, i, action);
  }

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
#ifndef BOARD_API_H
#define BOARD_API_H

#include "../include/_globals.h"

// ----------------------------------------
//  DATA TYPES
// ----------------------------------------

/** Ship ID enum
  - Used for indexing into the `remaining_ships` parameger of the `Board` struct (below)
*/
typedef enum ShipID : unsigned char {
  SHIP_PATROL_BOAT = 0,
  SHIP_SUBMARINE = 1,
  SHIP_DESTROYER = 2,
  SHIP_BATTLESHIP = 3,
  SHIP_CARRIER = 4
} ShipID;

/** Square enum
  - Provides the data for a given square on the board (details below)
*/
typedef enum Square : unsigned char {
  SQUARE_EMPTY = 0,
  SQUARE_MISS = 1,
  SQUARE_HIT = 2,
  SQUARE_SHIP_PATROL_BOAT = SHIP_PATROL_BOAT + SQUARE_HIT + 1,
  SQUARE_SHIP_SUBMARINE = SHIP_SUBMARINE + SQUARE_HIT + 1,
  SQUARE_SHIP_DESTROYER = SHIP_DESTROYER + SQUARE_HIT + 1,
  SQUARE_SHIP_BATTLESHIP = SHIP_BATTLESHIP + SQUARE_HIT + 1,
  SQUARE_SHIP_CARRIER = SHIP_CARRIER + SQUARE_HIT + 1
} Square;

/** Coordinate struct
  @param x - X coordinate on the board - Left hand side is 0, increases to the right
  @param y - Y coordinate on the board - Top is 0, increases downward
*/
typedef struct Coordinate {
  int x;
  int y;
} Coordinate;

/** Ship Placement struct
  @param x - (same as for Coordinate struct)
  @param y - (same as for Coordinate struct)
  @param rotation - Rotation of the ship [0-3]. Increasing -> counterclockwise
*/
typedef struct ShipPlacement {
  int x;
  int y;
  int rotation;
} ShipPlacement;

/** Action struct
  - Provides data on a specific action taken by the player, and its outcome
  @param pos - The coordinates of the action in question
  @param hit - The result of that shot
    - HitType : [SHOT_FAIL | SHOT_MISS | SHOT_HIT | SHOT_HIT_SUNK] (SHOT_FAIL will never appear in this context)
*/
typedef struct Action {
  Coordinate pos;
  HitType hit;
} Action;

/** Board struct
  - Provides the "player" view of the board

  @param board - 1D array representation of the 2D board - can be conveniently indexed into using the getSquare function
  below
    - For the players own board, shows hit/sunk markers, and also shows ships (full range of the `Square` enum)
    - For the opponent board, shows only hit/sunk markers (i.e. only `SQUARE_EMPTY`, `SQUARE_MISS`, `SQUARE_HIT` from
  the `Square` enum)
  @param remaining_ships - 1D array of the ships on the board - each ship's information is present at the index of its
  id. e.g. to retrieve data for the destroyer, get the value at `remaining_ships[SHIP_DESTROYER]`
    - For the players own board, this shows the number of remaining parts of the ship
    - For the opponent board, shows only whether the ship is still on the board (1) or has been sunk (0)
  @param history - A stack of all previous shots taken on this board, storing both position and hit/miss data
    - This can be used to determine when and where ships have been hit, as well as when they have been sunk
  @param history_head_ptr - The index of the current "head" of the stack
    - Can also be used as a turn counter - number of turns = history_head_ptr+1 (assuming no failed moves)
    - A value of -1 signifies that the stack is empty (i.e. first turn) - DO NOT read the history in this state: you
       will get erroneus data.
*/
typedef struct Board {
  Square board[BOARD_SIZE * BOARD_SIZE];
  char remaining_ships[(int)SHIP_CARRIER + 1];
  Action history[BOARD_SIZE * BOARD_SIZE];
  int history_head_ptr;
} Board;

// ----------------------------------------
//  FUNCTIONS
// ----------------------------------------

/** Check ship function
  - Checks whether the given ship type can be placed at the provided location and rotation
  @param board - The board the ship should be placed on (passed to the user function)
  @param ship - The ship ID (passed to the user function)
  @param placement - The position and rotation of the ship (as returned by the user function)

  @returns int [0|1]
  - 0 | Invalid position
  - 1 | Valid position
*/
int checkShip(Board* board, ShipID ship, ShipPlacement placement);

/** Check ship function
  - Checks whether the given position has already been shot at
  @param board - The board the shot is being taken on (passed to the user function)
  @param shot - The position of the shot (as returned by the user function)

  @returns int [0|1]
  - 0 | Invalid position
  - 1 | Valid position
*/
int checkShot(Board* board, Coordinate shot);

/** Helper function for checking a board square
  - This can be done yourself, but this function provides an abstracted method
*/
static Square getSquare(Board* board, int x, int y) { return board->board[y + x * BOARD_SIZE]; }

/** Helper function for checking a if a ship is still on the board (i.e. not sunk)
  - This can be done yourself, but this function provides an abstracted method
*/
static int checkShipExists(Board* board, ShipID ship) { return board->remaining_ships[ship]; }

/** Helper function for checking how many ships remain on the board
  - This can be done yourself, but this function provides an abstracted method
*/
static int currentShipCount(Board* board) {
  return (board->remaining_ships[0] != 0) + (board->remaining_ships[1] != 0) + (board->remaining_ships[2] != 0) +
         (board->remaining_ships[3] != 0) + (board->remaining_ships[4] != 0);
}

#endif
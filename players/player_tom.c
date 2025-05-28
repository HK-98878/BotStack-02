#include "../public/player.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static bool searched_up = false;
static bool searched_down = false;
static bool searched_left = false;
static bool searched_right = false;

static const int WEIGHTINGS[BOARD_SIZE][BOARD_SIZE] = {
    {9, 1, 5, 1, 5, 5, 1, 5, 1, 9},
    {1, 8, 1, 4, 2, 1, 4, 1, 8, 1},
    {5, 1, 7, 1, 3, 3, 1, 7, 1, 1},
    {1, 4, 1, 6, 1, 1, 6, 1, 4, 1},
    {5, 1, 3, 1, 5, 5, 1, 3, 2, 5},
    {5, 2, 3, 1, 5, 5, 1, 3, 1, 5},
    {1, 4, 1, 6, 1, 1, 6, 1, 1, 1},
    {5, 1, 7, 1, 3, 3, 1, 7, 1, 5},
    {1, 8, 1, 4, 1, 2, 4, 1, 8, 1},
    {9, 1, 5, 1, 5, 5, 1, 5, 1, 9}
};

static int dynamicWeightings[BOARD_SIZE][BOARD_SIZE];

static bool tracking_ship = false;
static Coordinate previousHit = {-1, -1};

// function definitions
Coordinate maxWeighting();
Coordinate getNextWeighting(Board board);
Coordinate trackRemainingShip(Board board, Coordinate pos);
void checkSurroundings(Board board, Coordinate coord);

ShipPlacement tomPlaceShip(Board board, ShipID ship) {
  switch (ship) {
    case SHIP_PATROL_BOAT:
      return (ShipPlacement){8, 9, 1};
    case SHIP_SUBMARINE:
      return (ShipPlacement){6, 8, 1};
    case SHIP_DESTROYER:
      return (ShipPlacement){8, 4, 2};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){5, 1, 1};
    case SHIP_CARRIER:
      return (ShipPlacement){2, 4, 0};
  }
}
Coordinate tomAction(Board board) {
  // return (Coordinate){-1, -1};
  Coordinate target = {-1, -1};

  if (board.history_head_ptr > -1) {
    Action last_hit = board.history[board.history_head_ptr];

    if (last_hit.hit == SHOT_HIT_SUNK) {
      // reset searching paramaters
      searched_up = false;
      searched_down = false;
      searched_left = false;
      searched_down = false;

      tracking_ship = false;
    }

    if (last_hit.hit == SHOT_HIT || tracking_ship) {
      if (last_hit.hit == SHOT_HIT) {
        previousHit = last_hit.pos;
      }

      target = trackRemainingShip(board, previousHit);

      tracking_ship = true;

      // if invalid position, continually add random offsets until a valid position is found
      while (!checkShot(&board, target)) {
        int num = rand() % 5;

        switch (num) {
          case 0:
            target.x = target.x + 1;
            break;
          case 1:
            target.x = target.x - 1;
            break;
          case 2:
            target.y = target.y + 1;
            break;
          case 3:
            target.y = target.y - 1;
            break;
        }
      }

      return target;
    }
  }

  // implementing choosing optional location if don't know any info
  target = getNextWeighting(board);

  // checkShot(&board, (Coordinate){x, y}));
  return target;
}

/// Trace rest of ship searching from up, down, left, right and returns position to shoot
Coordinate trackRemainingShip(Board board, Coordinate pos) {
  Coordinate shoot = {pos.x, pos.y};
  bool valid = false;

  int i = board.history_head_ptr;

  // if no free spaces surrounding current hit, backtrack to previous hits and then search again until the ship has been
  // completed
  while (!valid) {
    checkSurroundings(board, pos);

    if (searched_up && searched_down && searched_left && searched_right) {
      i--;
      if (i < 0) {
        break;
      }

      if (board.history[i].hit = SHOT_HIT) {
        pos = board.history[i].pos;
      }
    }
    else {
      valid = true;
    }

    if (i < 0) {
      valid = true;
    }
  }

  int prevIndex = board.history_head_ptr - 1;
  int prevIndex2 = board.history_head_ptr - 2;

  if (prevIndex > -1 && prevIndex2 > -1) {
    // if previous two shots in the same column, search up and down
    if (board.history[prevIndex].pos.x == board.history[prevIndex2].pos.x) {
      if (!searched_up) {
        shoot.y = pos.y - 1;
      }

      else if (!searched_down) {
        shoot.y = pos.y + 1;
      }

      return shoot;
    }

    // if previous two shots in the same row, search left and right
    if (board.history[prevIndex].pos.y == board.history[prevIndex2].pos.y) {
      if (!searched_left) {
        shoot.x = pos.x - 1;
      }

      else if (!searched_right) {
        shoot.x = pos.x + 1;
      }

      return shoot;
    }
  }

  // if have no knowledge on location of remaining ship, search up,down,left,right
  if (searched_up == false) {
    shoot.y = pos.y - 1;
  }
  else if (searched_down == false) {
    shoot.y = pos.y + 1;
  }
  else if (searched_left == false) {
    shoot.x = pos.x - 1;
  }
  else if (searched_right == false) {
    shoot.x = pos.x + 1;
  }

  return shoot;
}

Coordinate getNextWeighting(Board board) {
  Coordinate shoot = {-1, -1};
  bool valid = false;

  // if new match, reset dynamic weightings
  if (board.history_head_ptr == -1) {
    for (int i = 0; i < BOARD_SIZE; ++i) {
      memcpy(dynamicWeightings[i], WEIGHTINGS[i], sizeof(int) * BOARD_SIZE);
    }
  }

  while (!checkShot(&board, shoot)) {
    shoot = maxWeighting();

    dynamicWeightings[shoot.x][shoot.y] = -1;  // sets the weight to negative so no longer goes for the same target
  }

  // if any cases ignored, will choose random location that has not been shot already
  //  while (!valid){
  //    if (checkShot(&board, shoot) == 1){
  //      valid = true;
  //    }

  //   if (!valid){
  //     if (shoot.x < 0){
  //       shoot.x = rand() % 8;
  //     }
  //     if (shoot.y < 0){
  //       shoot.y = rand() % 8;
  //     }
  //   }
  // }

  return shoot;
}

Coordinate maxWeighting() {
  int max = dynamicWeightings[0][0];
  int x = 0;
  int y = 0;

  for (int i = 0; i < BOARD_SIZE; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      if (dynamicWeightings[i][j] > max) {
        max = dynamicWeightings[i][j];
        x = i;
        y = j;
      }
    }
  }

  Coordinate target = {x, y};

  return target;
}

void checkSurroundings(Board board, Coordinate coord) {
  searched_up = false;
  searched_down = false;
  searched_left = false;
  searched_right = false;

  // check if on border
  if (coord.x == 0) {
    searched_left = true;
  }
  if (coord.y == 0) {
    searched_up = true;
  }
  if (coord.x == 7) {
    searched_right = true;
  }
  if (coord.y == 7) {
    searched_down = true;
  }

  for (int i = 0; i <= board.history_head_ptr; i++) {
    if (board.history[i].pos.x == coord.x && board.history[i].pos.y == coord.y + 1) {
      searched_down = true;
    }

    if (board.history[i].pos.x == coord.x && board.history[i].pos.y == coord.y - 1) {
      searched_up = true;
    }

    if (board.history[i].pos.x == coord.x + 1 && board.history[i].pos.y == coord.y) {
      searched_right = true;
    }

    if (board.history[i].pos.x == coord.x - 1 && board.history[i].pos.y == coord.y) {
      searched_left = true;
    }
  }
}
#include "../public/player.h"

#include <stdio.h>
#include <stdlib.h>

ShipPlacement vinPlaceShip(Board board, ShipID ship) {
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

int onWall(int x, int y) {
  if (x == 9) {
    if (y == 0) {
      return 10;
    }
    else if (y == 9) {
      return 12;
    }
    else {
      return 1;
    }
  }
  else if (x == 0) {
    if (y == 0) {
      return 30;
    }
    else if (y == 9) {
      return 23;
    }
    else {
      return 3;
    }
  }
  else {
    if (y == 0) {
      return 0;
    }
    else if (y == 9) {
      return 2;
    }
    else {
      return -1;
    }
  }
}

Coordinate vinAction(Board board) {
  int x;
  int y;
  // do{
  static int checkCount;
  // static int checkCount;
  if (board.history == 0) {
    x = rand() % 10;
    y = rand() % 10;
    return (Coordinate){x, y};
    int checkCount = 0;
  }
  else {
    int xprev = board.history[board.history_head_ptr].pos.x;
    int yprev = board.history[board.history_head_ptr].pos.y;
    int hitORmiss = board.history[board.history_head_ptr].hit;
    if (hitORmiss == SHOT_HIT) {
      checkCount = 0;
    }

    int wall = onWall(xprev, yprev);  // call function
    // static int check = 0;

    if (hitORmiss == SHOT_HIT || checkCount == 1 || checkCount == 2 || checkCount == 3) {  // previous shot was
                                                                                           // succesful

      int bigSwitch;
      if (wall < 0) {
        bigSwitch = 0;
      }
      else if (wall < 10) {
        bigSwitch = 1;
      }
      else {
        bigSwitch = 2;
      }

      switch (bigSwitch) {
        case 0:
          // in the middle
          if (checkCount == 0) {
            x = xprev;
            y--;
          }
          else if (checkCount == 1) {
            x--;
            y = yprev;
          }
          else if (checkCount == 2) {
            x++;
            y = yprev;
          }
          else if (checkCount == 3) {
            x = xprev;
            y++;
          }

          checkCount++;
          break;

        case 1:
          switch (wall) {
            case 4:
              if (checkCount == 0) {
                x = xprev;
                y++;
              }
              else if (checkCount == 1) {
                y = yprev;
                x++;
              }
              else if (checkCount == 2) {
                y = yprev;
                x--;
              }
              break;
            case 1:
              if (checkCount == 0) {
                x--;
                y = yprev;
              }
              else if (checkCount == 1) {
                x = xprev;
                y++;
              }
              else if (checkCount == 2) {
                x = xprev;
                y--;
              }

              break;

            case 2:
              if (checkCount == 0) {
                x = xprev;
                y--;
              }
              else if (checkCount == 1) {
                x--;
                y = yprev;
              }
              else if (checkCount == 2) {
                x++;
                y = yprev;
              }
              break;

            case 3:
              if (checkCount == 0) {
                x++;
                y = yprev;
              }
              else if (checkCount == 1) {
                x = xprev;
                y++;
              }
              else if (checkCount == 2) {
                x = xprev;
                y--;
              }
              break;
              checkCount++;

              break;
            default:
          }
        case 2:

          switch (wall) {
            case 14:
              // check the one below and then the one on the left
              if (checkCount == 0) {
                x = xprev;
                y++;
              }
              else if (checkCount == 1) {
                y = yprev;
                x--;
              }

              break;
            case 34:
              // check the one below and the on the right
              if (checkCount == 0) {
                x = xprev;
                y++;
              }
              else if (checkCount == 1) {
                y = yprev;
                x++;
              }

              break;
            case 23:
              // check the one on top and the one on the right
              if (checkCount == 0) {
                x = xprev;
                y--;
              }
              else if (checkCount == 1) {
                y = yprev;
                x++;
              }
              break;
            case 12:
              // check the one on top and the one on the left
              if (checkCount == 0) {
                x = xprev;
                y--;
              }
              else if (checkCount == 1) {
                y = yprev;
                x--;
              }
              break;
            default:
          }
          checkCount++;
          break;
        default:
      }
    }
    else {  // previous shot missed
      x = rand() % 10;
      y = rand() % 10;
    }
  }
  //}while (!checkShot(&board, (Coordinate){x, y}));
  if (!checkShot(&board, (Coordinate){x, y})) {
    x = rand() % 10;
    y = rand() % 10;
  }
  return (Coordinate){x, y};
}
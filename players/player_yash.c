#include "../public/player.h"

// #include <stdio.h>
#include <stdlib.h>

static int searchList[50][2] = {
    {4, 4},
    {3, 5},
    {4, 6},
    {5, 7},
    {6, 6},
    {7, 5},
    {6, 4},
    {5, 3},
    {4, 2},
    {3, 3},
    {2, 4},
    {1, 5},
    {2, 6},
    {3, 7},
    {4, 8},
    {5, 9},
    {6, 8},
    {7, 7},
    {8, 6},
    {9, 5},
    {8, 4},
    {7, 3},
    {6, 2},
    {5, 1},
    {4, 0},
    {3, 1},
    {2, 2},
    {1, 3},
    {0, 4},
    {0, 6},
    {1, 7},
    {2, 8},
    {3, 9},
    {7, 9},
    {8, 8},
    {9, 7},
    {9, 3},
    {8, 2},
    {7, 1},
    {6, 0},
    {2, 0},
    {1, 1},
    {0, 2},
    {0, 8},
    {1, 9},
    {9, 9},
    {9, 1},
    {8, 0},
    {0, 0}
};

ShipPlacement yashPlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT:  // 2 squares long
      return (ShipPlacement){4, 5, 0};
    case SHIP_SUBMARINE:  // 3 squares long
      return (ShipPlacement){3, 0, 1};
    case SHIP_DESTROYER:  // 3 & 1 (L-shape)
      return (ShipPlacement){8, 2, 3};
    case SHIP_BATTLESHIP:  // 4 squares long
      return (ShipPlacement){8, 6, 2};
    case SHIP_CARRIER:  // 5 squares long
      return (ShipPlacement){1, 7, 0};
  }
}

Coordinate yashAction(Board board) {
  // int x = -1, y = -1;
  // do {
  //   x = rand() % 10, y = rand() % 10;
  // } while (!checkShot(&board, (Coordinate){x, y}));
  static int i = -1;
  int x, y;
  int searching = 1;
  int go_up = 0;
  int go_down = 0;
  int go_left = 0;
  int go_right = 0;
  int count_since_hit = 0;
  int first_down = 0;

  if (i == -1) {
    x = 5;
    y = 5;
    i++;
    return (Coordinate){x, y};
  }

  while (searching == 1) {
    Action past_result = board.history[board.history_head_ptr];
    if (past_result.hit == SHOT_HIT) {
      // printf("past hit\n");
      count_since_hit++;
      int ship_first_hit_x = past_result.pos.x;
      int ship_first_hit_y = past_result.pos.y;
      go_up = 1;
      break;
    }
    // printf("Searching");
    do {
      if (i <= 48) {
        x = searchList[i][0];
        y = searchList[i][1];
        i++;
      }
      else
        x = rand() % 10, y = rand() % 10;
      // printf("%d,%d \n", x, y);
    } while (!checkShot(&board, (Coordinate){x, y}));
    return (Coordinate){x, y};
  }

  // first try going up
  while (go_up == 1) {
    searching = 0;
    // printf("Trying up");
    Action past_up_result = board.history[board.history_head_ptr];
    if (past_up_result.hit == SHOT_MISS) {
      // printf("miss\n");
      go_up = 0;
      go_down = 1;
      break;
    }
    // keep going up if past shot was a hit
    else {
      int past_x = board.history[board.history_head_ptr].pos.x;
      int past_y = board.history[board.history_head_ptr].pos.y;

      // printf("Past x: %d, Past y: %d\n", past_x, past_y);
      int test_up = checkShot(&board, (Coordinate){past_x, past_y - 1});
      // printf("Have i shot up: %d \n", test_up);
      if (test_up == 1) {
        x = past_x;
        y = past_y - 1;
        count_since_hit++;
        return (Coordinate){x, y};
      }
      else {
        searching = 1;
        go_up = 0;
        do {
          x = searchList[i][0];
          y = searchList[i][1];
          // printf("%d,%d \n", x, y);
          i++;
        } while (!checkShot(&board, (Coordinate){x, y}));
        return (Coordinate){x, y};
      }
    }
  }

  // second try going down
  while (go_down == 1) {
    searching = 0;
    Action past_down_result = board.history[board.history_head_ptr];
    if (past_down_result.hit == SHOT_MISS) {
      // printf("miss\n");
      go_down = 0;
      go_left = 1;
      break;
    }
    // keep going down if past shot was a hit
    else {
      int past_x = board.history[board.history_head_ptr].pos.x;
      int past_y = board.history[board.history_head_ptr].pos.y;

      // printf("Past x: %d, Past y: %d\n", past_x, past_y);
      int test_down = checkShot(&board, (Coordinate){past_x, past_y});
      // printf("Have i shot down: %d \n", test_down);
      if (test_down == 0) {
        x = past_x;
        y = past_y + 1;
        return (Coordinate){x, y};
      }
      else {
        go_down = 0;
        go_left = 1;
      }
    }
  }

  // third try going left
  while (searching == 0 && go_left == 1) {
    Action past_left_result = board.history[board.history_head_ptr];
    if (past_left_result.hit == SHOT_MISS) {
      // printf("miss\n");
      go_left = 0;
      go_right = 1;
      break;
    }
    // keep going left if past shot was a hit
    else {
      int past_x = board.history[board.history_head_ptr].pos.x;
      int past_y = board.history[board.history_head_ptr].pos.y;

      // printf("Past x: %d, Past y: %d\n", past_x, past_y);
      int test_left = checkShot(&board, (Coordinate){past_x, past_y});
      // printf("Have i shot down: %d \n", test_left);
      if (test_left == 0) {
        x = past_x - 1;
        y = past_y;
        return (Coordinate){x, y};
      }
      else {
        go_left = 0;
        go_right = 1;
      }
    }
  }

  // fourth try going right
  while (searching == 0 && go_right == 1) {
    Action past_right_result = board.history[board.history_head_ptr];
    if (past_right_result.hit == SHOT_MISS) {
      // printf("miss\n");
      go_right = 0;
      go_up = 1;
      break;
    }
    // keep going right if past shot was a hit
    else {
      int past_x = board.history[board.history_head_ptr].pos.x;
      int past_y = board.history[board.history_head_ptr].pos.y;

      // printf("Past x: %d, Past y: %d\n", past_x, past_y);
      int test_right = checkShot(&board, (Coordinate){past_x, past_y});
      // printf("Have i shot down: %d \n", test_right);
      if (test_right == 0) {
        x = past_x + 1;
        y = past_y;
        return (Coordinate){x, y};
      }
      else {
        go_right = 0;
        go_up = 1;
      }
    }

    do {
      x = rand() % 10, y = rand() % 10;
    } while (!checkShot(&board, (Coordinate){x, y}));
  }

  // return (Coordinate){x, y};
}
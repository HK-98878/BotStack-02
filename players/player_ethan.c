#include "../public/player.h"

#include <stdlib.h>

int get_turn(Board board);
int get_x(Board board, int index);
int get_y(Board board, int index);
int get_hit(Board board, int index);

static int hit_turn;

typedef enum Direction : unsigned char { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3 } Direction;

int get_turn(Board board) {
  int turn = board.history_head_ptr + 1;
  return turn;
}

int get_x(Board board, int index) {
  int x_pos = board.history[board.history_head_ptr - index].pos.x;
  return x_pos;
}

int get_y(Board board, int index) {
  int y_pos = board.history[board.history_head_ptr - index].pos.y;
  return y_pos;
}

int get_hit(Board board, int index) {
  int hit_status = board.history[board.history_head_ptr - index].hit;
  return hit_status;
}

// int results = *malloc(get_turn(board) * sizeof(int));

ShipPlacement ethanPlaceShip(Board board, ShipID ship) {
  // return (ShipPlacement){-1, -1, -1};
  switch (ship) {
    case SHIP_PATROL_BOAT:
      return (ShipPlacement){2, 9, 0};
    case SHIP_SUBMARINE:
      return (ShipPlacement){3, 1, 0};
    case SHIP_DESTROYER:
      return (ShipPlacement){7, 2, 0};
    case SHIP_BATTLESHIP:
      return (ShipPlacement){1, 4, 0};
    case SHIP_CARRIER:
      return (ShipPlacement){6, 6, 1};
  }
}

Coordinate ethanAction(Board board) {
  // return (Coordinate){-1, -1};
  int x, y, i, j;
  int guesses[5] = {0, 2, 4, 6, 8};
  int mode = 0;
  int hit_status = SHOT_MISS;

  // Turn Status

  // Hit status
  hit_status = get_hit(board, 0);

  int turn = get_turn(board);
  // printf("Turn %d:\n", turn);

  if (hit_status == SHOT_HIT && turn != 0) {
    mode = 1;  // Attack mode
  }
  else {
    mode = 0;  // Hunt mode
  }

  // Better mode determination
  for (i = turn; i >= 0; i--) {
    if (get_hit(board, -(turn - i)) == SHOT_HIT)
      // printf("hit_status: %d\n", get_hit(board, -(turn-i)));
      for (j = i; j <= turn; j++) {
        if (get_hit(board, -(turn - j)) != SHOT_HIT_SUNK) {
          // printf("Attacking...\n");
          mode = 1;
          continue;
        }
        else {
          mode = 0;
          break;
        }
      }
  }

  // Mode Determination
  if (mode == 0 && turn < 25) {  // Hunt mode
    do {
      // printf("Hunt Mode \n");
      // Random guess in cross formation
      x = guesses[rand() % 5], y = guesses[rand() % 5];
      // printf("Y:%d,X:%d\n", y, x);

      if (turn > 25) {
        break;
      }

    } while (!checkShot(&board, (Coordinate){x, y}));
    return (Coordinate){x, y};
  }

  // If all even squares are already guessed
  if (mode == 0 && turn >= 25) {
    do {
      // printf("Hunt Mode \n");

      x = rand() % 10, y = rand() % 10;
    } while (!checkShot(&board, (Coordinate){x, y}));
    return (Coordinate){x, y};
  }

  if (mode == 1) {  // Attack mode
    // printf("Attack mode\n");
    int x_hit = get_x(board, 0);
    int y_hit = get_y(board, 0);

    int x_prev_hit;
    int y_prev_hit;

    if (get_hit(board, -1) == SHOT_HIT) {
      x_prev_hit = get_x(board, -1);
      y_prev_hit = get_y(board, -1);
    }
    // int *hit_guesses = (int *) malloc(sizeof(int) * 8);
    Direction options[4] = {LEFT, RIGHT, UP, DOWN};

    do {
      Direction direction = options[rand() % 4];

      switch (direction) {
        case 0:
          x = x_hit - 1;
          y = y_hit;
        case 1:
          x = x_hit + 1;
          y = y_hit;
        case 2:
          x = x_hit;
          y = y_hit + 1;
        case 3:
          x = x_hit;
          y = y_hit - 1;
        default:
          while (!checkShot(&board, (Coordinate){x, y})) {
            x = rand() % 10, y = rand() % 10;
          }
      }
    } while (x >= 0 && x < 10 && y >= 0 && y < 10 && x != x_prev_hit && y != y_prev_hit &&
             !checkShot(&board, (Coordinate){x, y}));

    return (Coordinate){x, y};
  }
}
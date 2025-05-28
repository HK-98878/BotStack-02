#include "board.h"

#include <stdio.h>

#define DISP_0 0x8000000000000000

static void rotate(int *x, int *y, int rot) {
  int temp;
  switch (rot) {
    case 0:  // No rotation
      break;
    case 1:  // 90 ccw
      temp = *x;
      *x = *y;
      *y = -temp;
      break;
    case 2:  // 180 ccw
      *x = -*x;
      *y = -*y;
      break;
    case 3:  // 270 ccw
      temp = *x;
      *x = -*y;
      *y = temp;
      break;
    default:  // Other
      break;
  }
}
static void rotate2(int *x, int *y, int rot) {
  int temp;
  switch (rot) {
    case 0:  // No rotation
      break;
    case 1:  // 90 ccw
      temp = *x;
      *x = -*y;
      *y = temp;
      break;
    case 2:  // 180 ccw
      *x = -*x;
      *y = -*y;
      break;
    case 3:  // 270 ccw
      temp = *x;
      *x = *y;
      *y = -temp;
      break;
    default:  // Other
      break;
  }
}

BoardData *initBoardData(int primaryPlayer) {
  BoardData *b = (BoardData *)calloc(1, sizeof(BoardData));
  b->primaryPlayer = primaryPlayer;
  b->history_head_ptr = -1;
#ifndef NO_GRAPHICS
  // Initialise the grid render
  b->board_render = tigrBitmap(BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
  tigrClear(b->board_render, SQ_COLOUR);
  for (int i = 0; i < BOARD_SIZE - 1; ++i) {
    tigrFill(b->board_render, (i + 1) * BOARD_SQUARE_SIZE - (BOARD_MARGIN_SIZE / 2), 0, 2,
             BOARD_RENDER_SIZE, BG_COLOUR);
    tigrFill(b->board_render, 0, (i + 1) * BOARD_SQUARE_SIZE - (BOARD_MARGIN_SIZE / 2),
             BOARD_RENDER_SIZE, 2, BG_COLOUR);
  }

  b->hit_squares_head = -1;
#endif

  return b;
}

#ifndef NO_GRAPHICS
void updateSquare(Tigr *board, int primaryPlayer, int x, int y, Square s) {
  x *= BOARD_SQUARE_SIZE;
  y *= BOARD_SQUARE_SIZE;
  TPixel c;
  switch (s) {
    case SQUARE_EMPTY:
      tigrFill(board, x, y, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, SQ_COLOUR);
      break;
    case SQUARE_HIT:
      // tigrBlitAlpha(board, hit, x, y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, 0xff);
      break;
    case SQUARE_MISS:
      tigrBlitTint(board, miss, x, y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
                   (TPixel){0x20, 0x20, 0x20, 0xff});
      // switch (backgroundState) {
      //   case 0:
      //     tigrBlitTint(board, miss, x, y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
      //     (TPixel){0x4e, 0x63, 0x71, 0xff}); break;
      //   case 1:
      //     tigrBlitTint(board, miss, x, y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
      //     (TPixel){0x2e, 0x54, 0x20, 0xff}); break;
      //   case 2:
      //     tigrBlitTint(board, miss, x, y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
      //     (TPixel){0xb1, 0x55, 0x2d, 0xff}); break;
      // }
      break;
    default:
      break;
  }
}
#endif

int placeShip(BoardData *board, ShipID shipId, int x, int y, int rot) {
  if (board == NULL)
    return 0;

  ShipData ship = ships[shipId];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, rot);
  x_place += x, y_place += y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, rot);

  if (OUT_OF_BOUNDS_WH(x_place, y_place, w_rot, h_rot))
    return 0;

  Coordinate *coords = calloc(ship.w * ship.h, sizeof(Coordinate));
  int ptr = -1;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, rot);

        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] == SQUARE_EMPTY)
          coords[++ptr] = (Coordinate){(x_place + j_rot), (y_place + i_rot)};
        else {
          free(coords);
          return 0;
        }
      }
    }
  }
  board->remaining_ship_squares[shipId] = ptr + 1;
  board->remaining_ships++;
  for (; ptr >= 0; --ptr)
    board->board[coords[ptr].y + coords[ptr].x * BOARD_SIZE] = SQUARE_SHIP_PATROL_BOAT + shipId;
#ifndef NO_GRAPHICS
  int sprite_size = BoatSpritesheetSizes[shipId];
  tigrBlitAlpha(
      board->board_render, boat_spritesheets[shipId], (x - (sprite_size / 2)) * BOARD_SQUARE_SIZE,
      (y - (sprite_size / 2)) * BOARD_SQUARE_SIZE, rot * (sprite_size * BOARD_SQUARE_SIZE), 0,
      sprite_size * BOARD_SQUARE_SIZE, sprite_size * BOARD_SQUARE_SIZE, 0xff);
#endif
  free(coords);

  return 1;
}
HitType shoot(BoardData *board, int x, int y, int *sunk) {
  *sunk = -1;
  if (OUT_OF_BOUNDS(x, y))
    return SHOT_FAIL;

  int i = y + x * BOARD_SIZE;
  switch (board->board[i]) {
    case SQUARE_HIT:
    case SQUARE_MISS:
      return SHOT_FAIL;
    case SQUARE_EMPTY:
      board->board[i] = SQUARE_MISS;
      board->history[++board->history_head_ptr] = (Action){
          (Coordinate){x, y},
          SHOT_MISS
      };
#ifndef NO_GRAPHICS
      updateSquare(board->board_render, board->primaryPlayer, x, y, SQUARE_MISS);
#endif
      return SHOT_MISS;
    case SQUARE_SHIP_PATROL_BOAT:
    case SQUARE_SHIP_SUBMARINE:
    case SQUARE_SHIP_DESTROYER:
    case SQUARE_SHIP_BATTLESHIP:
    case SQUARE_SHIP_CARRIER:
      board->remaining_ship_squares[board->board[i] - SQUARE_SHIP_PATROL_BOAT] -= 1;
#ifndef NO_GRAPHICS
      updateSquare(board->board_render, board->primaryPlayer, x, y, SQUARE_HIT);
      board->hit_squares[++board->hit_squares_head] = (Coordinate){x, y};
#endif

      if (board->remaining_ship_squares[board->board[i] - SQUARE_SHIP_PATROL_BOAT] == 0) {
        board->remaining_ships -= 1;
        *sunk = board->board[i] - SQUARE_SHIP_PATROL_BOAT;
        board->board[i] = SQUARE_HIT;
        board->history[++board->history_head_ptr] = (Action){
            (Coordinate){x, y},
            SHOT_HIT_SUNK
        };
        return SHOT_HIT_SUNK;
      }
      board->history[++board->history_head_ptr] = (Action){
          (Coordinate){x, y},
          SHOT_HIT
      };
      board->board[i] = SQUARE_HIT;
      return SHOT_HIT;
  }
}

#ifndef NO_GRAPHICS
void renderPlacementOverlay(BoardData *board, Tigr *render, int player1, ShipID shipId, int x,
                            int y, int rot) {
  if (board == NULL)
    return;

  ShipData ship = ships[shipId];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, rot);
  x_place += x, y_place += y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, rot);

  int ptr = -1, valid = 1;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, rot);

        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] != SQUARE_EMPTY)
          valid = 0;
        if (OUT_OF_BOUNDS(y_place + i_rot, x_place + j_rot))
          valid = 0;
      }
    }
  }
  board->remaining_ship_squares[shipId] = ptr + 1;
  board->remaining_ships += 1;

  TPixel tint = valid ? PLACEMENT_VALID_TINT : PLACEMENT_INVALID_TINT;
  int x_offset = TEXT_SPACE + (!player1) * (TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE);
  int sprite_size = BoatSpritesheetSizes[shipId];

  Tigr *frame = tigrBitmap(BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
  tigrClear(frame, (TPixel){0x00, 0x00, 0x00, 0x00});
  tigrBlitTint(frame, boat_spritesheets[shipId], (x - (sprite_size / 2)) * BOARD_SQUARE_SIZE,
               (y - (sprite_size / 2)) * BOARD_SQUARE_SIZE, rot * (sprite_size * BOARD_SQUARE_SIZE),
               0, sprite_size * BOARD_SQUARE_SIZE, sprite_size * BOARD_SQUARE_SIZE, tint);
  tigrBlitAlpha(render, frame, x_offset, TEXT_SPACE + HEADER_SPACE, 0, 0, BOARD_RENDER_SIZE,
                BOARD_RENDER_SIZE, 0xff);
}
#endif

// -------------------- API functions --------------------
int checkShip(Board *board, ShipID shipID, ShipPlacement placement) {
  if (board == NULL || shipID > SHIP_CARRIER)
    return 0;

  ShipData ship = ships[shipID];

  int x_place = -ship.w + ship.offset_x + 1, y_place = -ship.h + ship.offset_y + 1;
  rotate(&x_place, &y_place, placement.rotation);
  x_place += placement.x, y_place += placement.y;
  int w_rot = ship.w - 1, h_rot = ship.h - 1;
  rotate(&w_rot, &h_rot, placement.rotation);

  if (OUT_OF_BOUNDS_WH(x_place, y_place, w_rot, h_rot))
    return 0;

  int i_rot, j_rot;
  for (int i = 0; i < ship.h; ++i) {
    for (int j = 0; j < ship.w; ++j) {
      if (ship.disp & (DISP_0 >> (j + i * 8))) {
        i_rot = i, j_rot = j;
        rotate2(&i_rot, &j_rot, placement.rotation);

        if (board->board[(y_place + i_rot) + (x_place + j_rot) * BOARD_SIZE] != SQUARE_EMPTY)
          return 0;
      }
    }
  }
  return 1;
}
int checkShot(Board *board, Coordinate shot) {
  if (OUT_OF_BOUNDS(shot.x, shot.y))
    return 0;
  return board->board[shot.y + shot.x * BOARD_SIZE] != SQUARE_HIT &&
         board->board[shot.y + shot.x * BOARD_SIZE] != SQUARE_MISS;
}
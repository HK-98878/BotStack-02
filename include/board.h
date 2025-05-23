#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../public/board_api.h"
#include "_globals.h"

#ifndef NO_GRAPHICS
#include "tigr.h"
#endif

#define OUT_OF_BOUNDS(x, y)          x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE
#define OUT_OF_BOUNDS_WH(x, y, w, h) OUT_OF_BOUNDS(x, y) || OUT_OF_BOUNDS(x + w, y + h)

typedef struct ShipData {
  int w, h;
  int offset_x, offset_y;
  long long int disp;
} ShipData;

extern ShipData ships[];
extern int backgroundState;

typedef struct BoardData {
  Square board[BOARD_SIZE * BOARD_SIZE];
  unsigned char remaining_ship_squares[(int)SHIP_CARRIER + 1];
  unsigned char remaining_ships;
#ifndef NO_GRAPHICS
  Tigr* board_render;
  Coordinate hit_squares[18];
  int hit_squares_head;
#endif
  Action history[BOARD_SIZE * BOARD_SIZE];
  int history_head_ptr;

  int primaryPlayer;  // Used for rendering / obscuring in player mode
} BoardData;

BoardData* initBoardData(int primaryPlayer);
int placeShip(BoardData* board, ShipID shipId, int x, int y, int rot);
HitType shoot(BoardData* board, int x, int y, int* sunk);

#ifndef NO_GRAPHICS
void renderPlacementOverlay(BoardData* board, Tigr* render, int player1, ShipID shipId, int x, int y, int rot);
extern Tigr** boat_spritesheets;
extern Tigr *hit, *miss;
#endif

static Board* toBoard(BoardData* src, int obscure) {
  Board* board = (Board*)calloc(1, sizeof(Board));
  board->history_head_ptr = src->history_head_ptr;
  memcpy(board->history, src->history, sizeof(Action) * BOARD_SIZE * BOARD_SIZE);

  if (obscure) {
    // Only show hit/miss/empty
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
      switch (src->board[i]) {
        case SQUARE_EMPTY:
        case SQUARE_MISS:
        case SQUARE_HIT:
          board->board[i] = src->board[i];
          break;
        case SQUARE_SHIP_PATROL_BOAT:
        case SQUARE_SHIP_SUBMARINE:
        case SQUARE_SHIP_DESTROYER:
        case SQUARE_SHIP_BATTLESHIP:
        case SQUARE_SHIP_CARRIER:
          board->board[i] = SQUARE_EMPTY;
          break;
      }
    }

    // Only show which ships remain, not how many squares of them remain
    for (int i = 0; i <= SHIP_CARRIER; ++i) {
      board->remaining_ships[i] = src->remaining_ship_squares[i] > 0;
    }
  }
  else {
    memcpy(board->board, src->board, sizeof(Square) * BOARD_SIZE * BOARD_SIZE);
    memcpy(board->remaining_ships, src->remaining_ship_squares, sizeof(int) * SHIP_CARRIER + 1);
  }
  return board;
}

/* --------------------
  Animation data
-------------------- */
typedef enum AnimationState { ANIMATION_NONE, ANIMATION_SHOOT, ANIMATION_SPLASH, ANIMATION_HIT } AnimationState;

#define ANIMATION_DURATION_WATER     20
#define ANIMATION_DURATION_SHOOT_AIM 20
#define ANIMATION_DURATION_SHOOT     ANIMATION_DURATION_SHOOT_AIM + 10

#define FLAME_SPRITE_SIZE        100
#define FLAME_SPRITESHEET_WIDTH  4
#define ANIMATION_DURATION_FLAME 11

#define SPLASH_SPRITE_SIZE       100
#define SPLASH_SPRITESHEET_WIDTH 4

#define ANIMATION_DURATION_SPLASH       10
#define ANIMATION_DURATION_SPLASH_EXTRA 25
#define COLOUR_SPLASH_1 \
  (TPixel) { 0x12, 0x40, 0xff, 0xff }
#define COLOUR_SPLASH_2 \
  (TPixel) { 0x12, 0x60, 0xff, 0xff }

#define EXPLOSION_SPRITE_SIZE       100
#define EXPLOSION_SPRITESHEET_WIDTH 4

#define ANIMATION_DURATION_HIT       11
#define ANIMATION_DURATION_HIT_EXTRA 25
#define COLOUR_HIT_1 \
  (TPixel) { 0xff, 0x90, 0x00, 0xff }
#define COLOUR_HIT_2 \
  (TPixel) { 0xff, 0xdd, 0x00, 0xff }

static double ease_out(int t_start, int t_end, double start, double end, int t) {
  if (t < t_start)
    return start;
  if (t > t_end)
    return end;

  double x = (((double)t_end - t) / (t_end - t_start));
  return (1 - (x * x)) * (end - start) + start;
}
static double ease_in(int t_start, int t_end, double start, double end, int t) {
  if (t < t_start)
    return start;
  if (t > t_end)
    return end;

  double x = (((double)t_end - t) / (t_end - t_start));
  return x * x * (end - start) + start;
}

#endif
#ifndef GLOBALS
#define GLOBALS

#ifdef PYTHON_BOT
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PLAYER_FILE "player1.py"
#endif

#define BOARD_SIZE 10

#define BOARD_SQUARE_SIZE 100
#define BOARD_MARGIN_SIZE 2
// #define BOARD_RENDER_SIZE ((BOARD_SIZE * BOARD_SQUARE_SIZE) + ((BOARD_SIZE - 1) * BOARD_MARGIN_SIZE))
#define BOARD_RENDER_SIZE (BOARD_SIZE * BOARD_SQUARE_SIZE)

#define WATER_SPRITE_SIZE_X 1432

#define PATROL_BOAT_SPRITE_SIZE 3
#define SUBMARINE_SPRITE_SIZE   3
#define DESTROYER_SPRITE_SIZE   3
#define BATTLESHIP_SPRITE_SIZE  5
#define CARRIER_SPRITE_SIZE     5
const extern int BoatSpritesheetSizes[5];

#define ICON_PATROL_BOAT_WIDTH 164
#define ICON_SUBMARINE_WIDTH   240
#define ICON_DESTROYER_WIDTH   160
#define ICON_BATTLESHIP_WIDTH  185
#define ICON_CARRIER_WIDTH     250
#define ICON_HEIGHT            200
const extern int IconWidths[5];

#define TEXT_SPACE 40
#define ICON_SPACE 250

#define SHOT_R 5

#ifdef NO_FRAME_DELAY
#define FRAME_TIME 0
#else
#define FRAME_TIME 50  // 20 FPS
#endif

#define BG_COLOUR \
  (TPixel) { 0x0f, 0x0f, 0x0f, 0xff }
#define SQ_COLOUR \
  (TPixel) { 0x00, 0x00, 0x00, 0x00 }
#define PLACEMENT_VALID_TINT \
  (TPixel) { 0xff, 0xff, 0xff, 0x8f }
#define PLACEMENT_INVALID_TINT \
  (TPixel) { 0xff, 0x00, 0x00, 0x8f }
#define SHIP_COLOUR \
  (TPixel) { 0x8f, 0x8f, 0x8f, 0xff }

#define SHIP_DEAD_COLOUR \
  (TPixel) { 0x1f, 0x1f, 0x1f, 0xff }
#define SHIP_LIVE_COLOUR \
  (TPixel) { 0xff, 0xff, 0xff, 0xff }

#define WATER_TINT \
  (TPixel) { 0xcf, 0xcf, 0xff, 0xff }

typedef enum HitType { SHOT_FAIL = -1, SHOT_MISS = 0, SHOT_HIT = 1, SHOT_HIT_SUNK = 2 } HitType;

// Helper function for player input
static int toCoords(const char* input, int* x, int* y) {
  *x = -1, *y = -1;

  if (input[0] == '0')
    return 0;

  if (!(input[0] >= 'A' && input[0] <= 'A' + BOARD_SIZE - 1) && !(input[0] >= 'a' && input[0] <= 'a' + BOARD_SIZE - 1))
    return 0;
  *x = (input[0] >= 'a' ? input[0] - 32 : input[0]) - 'A';

  for (int i = 1; i < 4; ++i) {
    if (input[i] == 0) {
      *y -= 1;
      return *y != -1;
    }

    if (input[i] >= '0' && input[i] <= '9')
      *y = *y == -1 ? (input[i] - '0') : (*y * 10 + (input[i] - '0'));
  }
  return 0;
}

#endif
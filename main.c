// Included first due to the python header needing to be first
#include "include/_globals.h"
// These are included as part of the Python include, so only need to be included if this isn't being used
#ifndef PYTHON_BOT
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#endif

#if defined(SOUND) && defined(_WIN32)
#include <windows.h>
#pragma comment(lib, "winmm.lib")  // This adds the static library "winmm.lib" to the project

const char explosions[4][50] = {"resources/sounds/explosion_1.wav", "resources/sounds/explosion_2.wav",
                                "resources/sounds/explosion_3.wav", "resources/sounds/explosion_4.wav"};
const char splashes[4][50] = {"resources/sounds/splash_1.wav", "resources/sounds/splash_2.wav",
                              "resources/sounds/splash_3.wav", "resources/sounds/splash_4.wav"};
#elif defined(SOUND)
#undef SOUND
#endif

// Game logic headers
#include "include/board.h"

// Rendering headers
#ifndef NO_GRAPHICS
#include "include/tigr_resize.h"
#endif

// Player headers
#include "public/player1.h"
#include "public/player2.h"

#define ERROR_MSG(msg)    \
  printf("[ERROR] " msg); \
  return 1;

#if defined(LOG) && LOG != 1
#define LOG_LOCATION f
FILE* f;
#else
#define LOG_LOCATION stdout
#endif

ShipData ships[5] = {
    (ShipData){1, 2, 0, 0, 0x8080000000000000},
    (ShipData){1, 3, 0, 1, 0x8080800000000000},
    (ShipData){2, 3, 1, 1, 0x8080C00000000000},
    (ShipData){1, 4, 0, 2, 0x8080808000000000},
    (ShipData){1, 5, 0, 2, 0x8080808080000000}
};
#ifndef NO_GRAPHICS
TigrFont* old_london_25;
Tigr *explosion, *splash, *target, *miss, *fire_anim, *boat_icons;
Tigr *water_spritesheet_blue, *water_spritesheet_red, *water_spritesheet_green;
int backgroundState = 0;

Tigr** boat_spritesheets;
const int BoatSpritesheetSizes[5] = {PATROL_BOAT_SPRITE_SIZE, SUBMARINE_SPRITE_SIZE, DESTROYER_SPRITE_SIZE,
                                     BATTLESHIP_SPRITE_SIZE, CARRIER_SPRITE_SIZE};
const int IconWidths[5] = {ICON_PATROL_BOAT_WIDTH, ICON_SUBMARINE_WIDTH, ICON_DESTROYER_WIDTH, ICON_BATTLESHIP_WIDTH,
                           ICON_CARRIER_WIDTH};
int fire_frame_offsets[36];

int shootAnim(Tigr* game, BoardData* b, int player1, int x, int y);

void renderBackground(Tigr* game, Tigr* background);
void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2);
void updateGameOverlays(Tigr* game, BoardData* p1, BoardData* p2);
void renderLabels(Tigr* game);
void renderIcons(Tigr* game);
void renderDeath(Tigr* game, int i);

#define LOAD_IMAGE(img, path)            \
  img = tigrLoadImage(path);             \
  if (img == NULL) {                     \
    printf("Failed to load [" path "]"); \
    return 0;                            \
  }
#endif

#define CHECK_END_GAME(p, player)                      \
  if (p->remaining_ships == 0) {                       \
    fprintf(LOG_LOCATION, "Player " player " wins\n"); \
    flags |= GAME_OVER;                                \
  }

static Coordinate toRender(int x, int y, int board) {
  return (Coordinate){
      TEXT_SPACE + (x * BOARD_SQUARE_SIZE) + (board == 2) * (BOARD_RENDER_SIZE + TEXT_SPACE + BOARD_SQUARE_SIZE),
      TEXT_SPACE + (y * BOARD_SQUARE_SIZE),
  };
}

#define BUFFER 4

#define PLAYER_1     0b00000001
#define SHIPS_PLACED 0b00000010
#define FIRST_RENDER 0b00000100
#define GAME_OVER    0b00001000
#define SHOOT        0b00010000

AnimationState anim_state = ANIMATION_NONE;
int anim_time = 0;
Coordinate anim_pos;

#ifdef PYTHON_BOT
PyObject* placeFnc;
PyObject* actionFnc;
#endif

int main() {
#ifdef PYTHON_BOT
  // Initialise python and add the current directory to sys.path
  Py_Initialize();
  // PyObject* sysmodule = PyImport_ImportModule("sys");
  // PyObject* syspath = PyObject_GetAttrString(sysmodule, "path");
  // PyList_Append(syspath, PyBytes_FromString("."));
  // Py_DECREF(syspath), Py_DECREF(sysmodule);

  FILE* exp_file;
  PyObject *main_module, *global_dict;
  // Load the relevant functions from the file
  exp_file = fopen(PLAYER_FILE, "r");
  PyRun_SimpleFile(exp_file, PLAYER_FILE);

  main_module = PyImport_AddModule("__main__");
  global_dict = PyModule_GetDict(main_module);

  char* exec_str = calloc(500, sizeof(char));
  sprintf(exec_str,
          "BOARD_SIZE=%d\nSHIP_PATROL_BOAT=%d\nSHIP_SUBMARINE=%d\nSHIP_DESTROYER=%d\nSHIP_BATTLESHIP="
          "%d\nSHIP_CARRIER=%d\nSQUARE_EMPTY=%d\nSQUARE_MISS=%d\nSQUARE_HIT=%d\nSQUARE_SHIP_PATROL_BOAT="
          "%d\nSQUARE_SHIP_SUBMARINE=%d\nSQUARE_SHIP_DESTROYER=%d\nSQUARE_SHIP_BATTLESHIP="
          "%d\nSQUARE_SHIP_CARRIER=%d\nSHOT_MISS=%d\nSHOT_HIT=%d\nSHOT_SUNK=%d\n",
          BOARD_SIZE, SHIP_PATROL_BOAT, SHIP_SUBMARINE, SHIP_DESTROYER, SHIP_BATTLESHIP, SHIP_CARRIER, SQUARE_EMPTY,
          SQUARE_MISS, SQUARE_HIT, SQUARE_SHIP_PATROL_BOAT, SQUARE_SHIP_SUBMARINE, SQUARE_SHIP_DESTROYER,
          SQUARE_SHIP_BATTLESHIP, SQUARE_SHIP_CARRIER, SHOT_MISS, SHOT_HIT, SHOT_HIT_SUNK);
  PyRun_SimpleString(exec_str);
  free(exec_str), exec_str = NULL;

  placeFnc = PyDict_GetItemString(global_dict, "player1PlaceShip");
  actionFnc = PyDict_GetItemString(global_dict, "player1Action");
  if (actionFnc == NULL || placeFnc == NULL) {
    ERROR_MSG("Failed to load Python functions")
  }
#endif

#if defined(LOG) && LOG != 1
  f = fopen(LOG, "ab+");
#endif

  srand(time(0));
#ifndef NO_GRAPHICS
  // ---------- Load resources ----------
  Tigr* font_img = tigrLoadImage("resources/OldLondon_25.png");
  if (font_img == NULL) {
    ERROR_MSG("Failed to load font image")
  }
  old_london_25 = tigrLoadFont(font_img, TCP_1252);
  if (old_london_25 == NULL) {
    ERROR_MSG("Failed to load font")
  }

  LOAD_IMAGE(water_spritesheet_blue, "resources/_Spritesheet_Water_Blue.png");
  LOAD_IMAGE(water_spritesheet_green, "resources/_Spritesheet_Water_Green.png");
  LOAD_IMAGE(water_spritesheet_red, "resources/_Spritesheet_Water_Red.png");
  LOAD_IMAGE(boat_icons, "resources/_Spritesheet__Boat_Icons.png");
  LOAD_IMAGE(explosion, "resources/_Animation_Explosion.png");
  LOAD_IMAGE(splash, "resources/_Animation_Splash.png");
  LOAD_IMAGE(fire_anim, "resources/_Animation_Flame.png");
  LOAD_IMAGE(target, "resources/Target.png");
  LOAD_IMAGE(miss, "resources/Miss.png");

  boat_spritesheets = calloc(5, sizeof(Tigr*));
  LOAD_IMAGE(boat_spritesheets[0], "resources/_Spritesheet_PatrolBoat.png")
  LOAD_IMAGE(boat_spritesheets[1], "resources/_Spritesheet_Submarine.png")
  LOAD_IMAGE(boat_spritesheets[2], "resources/_Spritesheet_Destroyer.png")
  LOAD_IMAGE(boat_spritesheets[3], "resources/_Spritesheet_Battleship.png")
  LOAD_IMAGE(boat_spritesheets[4], "resources/_Spritesheet_Carrier.png")

  for (int i = 0; i < 36; ++i) {
    fire_frame_offsets[i] = rand() % ANIMATION_DURATION_FLAME;
  }

  // ---------- Window setup ----------
  Tigr* game = tigrBitmap(2 * BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                          TEXT_SPACE + BOARD_RENDER_SIZE + ICON_SPACE);
  tigrClear(game, BG_COLOUR);
  game->blitMode = TIGR_KEEP_ALPHA;
  renderLabels(game);
  renderIcons(game);

  TigrResize* window = (TigrResize*)calloc(1, sizeof(TigrResize));
  window->window = tigrWindow(game->w, game->h, "Game", TIGR_AUTO);
  window->contents = game;
#endif

  // ---------- Game variables ----------
  // Gamestate
  unsigned char flags = PLAYER_1;
#ifdef PLAYER
  ShipID playerCurrentShip = SHIP_PATROL_BOAT;
#endif

  // Function pointers
  ShipPlacement (*p1PlaceFuncPtr)(Board, ShipID) = &player1PlaceShip;
  Coordinate (*p1ActionFuncPtr)(Board) = &player1Action;
  ShipPlacement (*p2PlaceFuncPtr)(Board, ShipID) = &player2PlaceShip;
  Coordinate (*p2ActionFuncPtr)(Board) = &player2Action;

  // ---------- Player setup ----------
#if defined(NO_GRAPHICS) || defined(LOG)
  fprintf(LOG_LOCATION, "----------\n GAME START\n----------\n\n");
#endif

  BoardData* p1 = initBoardData(0);
  BoardData* p2 = initBoardData(1);

  ShipPlacement placement;
  Board* b;
  Coordinate shot;
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p1, 0);
    placement = (*p1PlaceFuncPtr)(*b, i);
    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR_MSG("Player 1 - Undefined ship placement function")
    }
    if (!placeShip(p1, i, placement.x, placement.y, placement.rotation)) {
      ERROR_MSG("Player 1 - Invalid ship placement")
    }
#if defined(NO_GRAPHICS) || defined(LOG)
    else
      fprintf(LOG_LOCATION, "[Player 1] Place %d | %d %d - %d\n", i, placement.x, placement.y, placement.rotation);
#endif
  }

  char s[BUFFER];
  int str_ptr = 0, c, sunk;
  memset(s, 0, BUFFER);

#ifndef PLAYER
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p2, 0);
    placement = (*p2PlaceFuncPtr)(*b, i);
    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR_MSG("Player 2 - Undefined ship placement function")
    }
    if (!placeShip(p2, i, placement.x, placement.y, placement.rotation)) {
      ERROR_MSG("Player 2 - Invalid ship placement")
    }
#if defined(NO_GRAPHICS) || defined(LOG)
    else
      fprintf(LOG_LOCATION, "[Player 2] Place %d | %d %d - %d\n", i, placement.x, placement.y, placement.rotation);
#endif
  }
  flags |= SHIPS_PLACED;
#elif defined(NO_GRAPHICS)
  int rotation, x, y;
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    printf("Enter the position for ship %d: ", i + 1);
    c = 0;
    fseek(stdin, 0, SEEK_END);
    while (c != '\n' && c != '\r') {
      c = getchar();
      s[str_ptr++] = c;
      if (str_ptr == BUFFER) {
        memset(s, 0, BUFFER);
        str_ptr = 0;
      }
    }
    toCoords(s, &x, &y);
    if (x == -1 || y == -1) {
      printf("Invalid input\n");
      --i;
      memset(s, 0, BUFFER);
      str_ptr = 0;
      continue;
    }
    printf("Enter the rotation [0-3]: ");
    fseek(stdin, 0, SEEK_END);
    c = getchar();
    if (c < '0' || c > '3') {
      printf("Invalid input\n");
      --i;
    }
    else {
      rotation = c - '0';
      if (!placeShip(p2, i, x, y, rotation)) {
        printf("Invalid ship placement\n");
        --i;
      }
      else
        fprintf(LOG_LOCATION, "[Player 2] Place %d | %d %d - %d\n", i, x, y, rotation);
    }

    memset(s, 0, BUFFER);
    str_ptr = 0;
  }
#else
  placement = (ShipPlacement){5, 5, 0};
#endif

  clock_t t, last_t = clock();
#ifndef NO_GRAPHICS
  while (tigrReadChar(window->window) != 0)
    ;
#endif
  while (
#ifndef NO_GRAPHICS
      !tigrClosed(window->window)
#else
      1
#endif
  ) {
    t = clock();
    if (t - last_t < FRAME_TIME)
      continue;
    last_t = t;

    // Manage player manual ship placement
#if defined(PLAYER) && !defined(NO_GRAPHICS)
    if (!(flags & SHIPS_PLACED)) {
      if (!(flags & FIRST_RENDER)) {
        flags |= FIRST_RENDER;

        updateGameRender(game, p1, p2);
        renderPlacementOverlay(p2, game, 0, playerCurrentShip, placement.x, placement.y, placement.rotation);
      }

      renderBackground(game, water_spritesheet_blue);
      updateGameRender(game, p1, p2);
      renderPlacementOverlay(p2, game, 0, playerCurrentShip, placement.x, placement.y, placement.rotation);

      c = tigrReadChar(window->window);
      if (c != 0) {
        switch (c) {
          case 'a':
            if (placement.x > 0)
              --placement.x;
            break;
          case 'd':
            if (placement.x < BOARD_SIZE - 1)
              ++placement.x;
            break;
          case 'w':
            if (placement.y > 0)
              --placement.y;
            break;
          case 's':
            if (placement.y < BOARD_SIZE - 1)
              ++placement.y;
            break;
          case ' ':
            placement.rotation = (placement.rotation + 1) % 4;
            break;
          case '\n':
          case '\r':
            if (placeShip(p2, playerCurrentShip, placement.x, placement.y, placement.rotation) == 1) {
              playerCurrentShip++;
              placement = (ShipPlacement){5, 5, 0};
              if (playerCurrentShip > SHIP_CARRIER)
                flags |= SHIPS_PLACED;
            }
            break;
        }
        if (playerCurrentShip <= SHIP_CARRIER)
          renderPlacementOverlay(p2, game, 0, playerCurrentShip, placement.x, placement.y, placement.rotation);
      }
      tigrResizeUpdate(window);
      continue;
    }
#endif

    if (anim_state == ANIMATION_NONE) {
      if (flags & PLAYER_1) {
        b = toBoard(p2, 1);
        shot = (*p1ActionFuncPtr)(*b);
        free(b), b = NULL;
        if (shot.x == -1 && shot.y == -1) {
          ERROR_MSG("Player 1 - Unimplemented shot function")
        }
#if defined(NO_GRAPHICS)
        fprintf(LOG_LOCATION, "[Player 1] Shoot   | %d %d\n", shot.x, shot.y);

        if (!shoot(p2, shot.x, shot.y, &sunk))
          flags &= ~PLAYER_1;
        else {
          fprintf(LOG_LOCATION, "\t Hit");
          if (sunk != -1)
            fprintf(LOG_LOCATION, " - Sunk (%d)\n", sunk);
          else
            fprintf(LOG_LOCATION, "\n");
        }
        CHECK_END_GAME(p2, "1")
#else
        anim_state = ANIMATION_SHOOT, anim_time = 0;
        anim_pos = toRender(shot.x, shot.y, 2);
        flags |= SHOOT;
#if defined(SOUND) && !defined(NO_ANIM)
        PlaySound("resources/sounds/shoot.wav", NULL, SND_ASYNC);
#endif
#endif
      }

      else {
#if defined(PLAYER) && !defined(NO_GRAPHICS)
        c = tigrReadChar(window->window);
        if (c != 0) {
          if (c == '\n' || c == '\r') {
            s[str_ptr] = 0;
            if (toCoords(s, &shot.x, &shot.y)) {
              anim_state = ANIMATION_SHOOT, anim_time = 0;
              anim_pos = toRender(shot.x, shot.y, 1);
              flags |= SHOOT;
#if defined(SOUND) && !defined(NO_ANIM)
              PlaySound("resources/sounds/shoot.wav", NULL, SND_ASYNC);
#endif
            }
            memset(s, 0, BUFFER);
            str_ptr = 0;
          }
          else {
            s[str_ptr++] = c;
            if (str_ptr == BUFFER) {
              memset(s, 0, BUFFER);
              str_ptr = 0;
            }
          }
        }
#elif defined(PLAYER)
        // & defined(NO_GRAPHICS)
        printf("Previous shot: (%d,%d) | %s\n", p1->history[p1->history_head_ptr].pos.x,
               p1->history[p1->history_head_ptr].pos.y,
               p1->history[p1->history_head_ptr].hit == SHOT_HIT_SUNK ? "Sunk"
               : p1->history[p1->history_head_ptr].hit == SHOT_HIT    ? "Hit"
                                                                      : "Miss");
        while (1) {
          printf(" Enter your target: ");
          scanf_s("%4s", s);
          if (!toCoords(s, &x, &y)) {
            printf("Invalid input\n");
            fseek(stdin, 0, SEEK_END);
            continue;
          }
          break;
        }
        sprintf(stdout, "[Player 2] Shoot   | %d %d\n", x, y);
        if (!shoot(p1, x, y, &sunk))
          flags |= PLAYER_1;
        else {
          fprintf(LOG_LOCATION, "\t Hit");
          if (sunk != -1)
            fprintf(stdout, " - Sunk (%d)\n", sunk);
          else
            fprintf(stdout, "\n");
        }
        CHECK_END_GAME(p1, "2")
#else
        b = toBoard(p1, 1);
        shot = (*p2ActionFuncPtr)(*b);
        free(b), b = NULL;
        if (shot.x == -1 && shot.y == -1) {
          ERROR_MSG("Player 2 - Unimplemented shot function")
        }
#ifdef NO_GRAPHICS
        fprintf(LOG_LOCATION, "[Player 2] Shoot   | %d %d\n", shot.x, shot.y);
        if (!shoot(p1, shot.x, shot.y, &sunk))
          flags |= PLAYER_1;
        else {
          fprintf(LOG_LOCATION, "\t Hit");
          if (sunk != -1)
            fprintf(LOG_LOCATION, " - Sunk (%d)\n", sunk);
          else
            fprintf(LOG_LOCATION, "\n");
        }
        CHECK_END_GAME(p1, "2")
#else
        anim_state = ANIMATION_SHOOT, anim_time = 0;
        anim_pos = toRender(shot.x, shot.y, 1);
        flags |= SHOOT;
#if defined(SOUND) && !defined(NO_ANIM)
        PlaySound("resources/sounds/shoot.wav", NULL, SND_ASYNC);
#endif
#endif

#endif
      }
    }

#ifndef NO_GRAPHICS
#ifdef NO_ANIM
    anim_state = ANIMATION_NONE;
#endif
    Tigr** bg;
    switch (backgroundState) {
      case 0:
        bg = &water_spritesheet_blue;
        break;
      case 1:
        bg = &water_spritesheet_green;
        break;
      case 2:
        bg = &water_spritesheet_red;
        break;
    }
    renderBackground(game, *bg);
    updateGameRender(game, p1, p2);
    updateGameOverlays(game, p1, p2);
    if (anim_state == ANIMATION_NONE && flags & SHOOT) {
      flags &= ~SHOOT;
      if (flags & PLAYER_1) {
#ifdef LOG
        fprintf(LOG_LOCATION, "[Player 1] Shoot   | %d %d\n", shot.x, shot.y);
#endif
        if (!shootAnim(game, p2, 0, shot.x, shot.y))
          flags &= ~PLAYER_1;
        CHECK_END_GAME(p2, "1")
      }
      else {
#ifdef LOG
        fprintf(LOG_LOCATION, "[Player 2] Shoot   | %d %d\n", shot.x, shot.y);
#endif
        if (!shootAnim(game, p1, 1, shot.x, shot.y))
          flags |= PLAYER_1;

        CHECK_END_GAME(p1, "2")
      }
    }
    tigrResizeUpdate(window);
    if (anim_state == ANIMATION_NONE && flags & GAME_OVER)
      break;
#else
    if (flags & GAME_OVER)
      break;
#endif
  }

#ifndef NO_GRAPHICS
  tigrFree(water_spritesheet_blue);
  tigrFree(game);
  tigrFree(window->window);
  if (window->contents_display != NULL)
    tigrFree(window->contents_display);
  free(window);
#endif

#ifdef PYTHON_BOT
  if (Py_FinalizeEx() < 0) {
    exit(120);
  }
  fclose(exp_file);
#endif

#ifdef LOG
  fclose(LOG_LOCATION);
#endif

  return 0;
}

#ifndef NO_GRAPHICS
int shootAnim(Tigr* game, BoardData* b, int player1, int x, int y) {
  int sunk, select;
  switch (shoot(b, x, y, &sunk)) {
    case SHOT_FAIL:
      return 0;
    case SHOT_HIT:
      anim_state = ANIMATION_HIT;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
#ifdef LOG
      fprintf(LOG_LOCATION, "\t Hit\n");
#endif
#if defined(SOUND) && !defined(NO_ANIM)
      select = rand() % 4;
      PlaySound(explosions[select], NULL, SND_ASYNC);
#endif
      return 1;
    case SHOT_HIT_SUNK:
      anim_state = ANIMATION_HIT;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
      renderDeath(game, (player1 ? 0 : 5) + sunk);
#ifdef LOG
      fprintf(LOG_LOCATION, "\t Hit - Sunk (%d)\n", sunk);
#endif
#if defined(SOUND) && !defined(NO_ANIM)
      select = rand() % 4;
      PlaySound(explosions[select], NULL, SND_ASYNC);
#endif
      return 1;
    case SHOT_MISS:
      anim_state = ANIMATION_SPLASH;
      anim_pos = toRender(x, y, player1 ? 1 : 2);
#if defined(SOUND) && !defined(NO_ANIM)
      select = rand() % 4;
      PlaySound(splashes[select], NULL, SND_ASYNC);
#endif
      return 0;
  }
}

void renderBackground(Tigr* game, Tigr* background) {
  static int frame = 0, frame_offset = 0;
  static signed char frame_offset_dir = 1;

  tigrBlit(game, background, TEXT_SPACE, TEXT_SPACE, frame_offset + (frame * WATER_SPRITE_SIZE_X), 0, BOARD_RENDER_SIZE,
           BOARD_RENDER_SIZE);
  tigrBlit(game, background, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE, TEXT_SPACE,
           frame_offset + (frame)*WATER_SPRITE_SIZE_X, 0, BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);

  frame = (frame + 1) % ANIMATION_DURATION_WATER;
  if (frame_offset == WATER_SPRITE_SIZE_X - BOARD_RENDER_SIZE)
    frame_offset_dir = -1;
  else if (frame_offset == 0)
    frame_offset_dir = 1;
  frame_offset = (frame_offset + (2 * frame_offset_dir));
}

void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2) {
  tigrBlitAlpha(game, p1->board_render, TEXT_SPACE, TEXT_SPACE, 0, 0, BOARD_RENDER_SIZE, BOARD_RENDER_SIZE, 0xff);
  tigrBlitAlpha(game, p2->board_render, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE, TEXT_SPACE, 0, 0,
                BOARD_RENDER_SIZE, BOARD_RENDER_SIZE, 0xff);
}
void updateGameAnimations(Tigr* game) {
  if (anim_state == ANIMATION_SHOOT) {
    if (anim_time < ANIMATION_DURATION_SHOOT_AIM) {
      tigrBlitTint(game, target, anim_pos.x, anim_pos.y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE,
                   (TPixel){0xff, 0xff, 0xff, 0xff / ANIMATION_DURATION_SHOOT_AIM * anim_time});
    }
    else {
      int r = ease_in(ANIMATION_DURATION_SHOOT_AIM, ANIMATION_DURATION_SHOOT, SHOT_R,
                      (int)((BOARD_SQUARE_SIZE - 1) / 2), anim_time);
      tigrBlitAlpha(game, target, anim_pos.x, anim_pos.y, 0, 0, BOARD_SQUARE_SIZE, BOARD_SQUARE_SIZE, 0xff);
      tigrFillCircle(game, anim_pos.x + BOARD_SQUARE_SIZE / 2, anim_pos.y + BOARD_SQUARE_SIZE / 2, r,
                     (TPixel){0x00, 0x00, 0x00, 0xff});
    }
    if (anim_time++ == ANIMATION_DURATION_SHOOT) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
  else if (anim_state == ANIMATION_SPLASH) {
    if (anim_time < ANIMATION_DURATION_SPLASH)
      tigrBlitAlpha(game, splash, anim_pos.x, anim_pos.y, SPLASH_SPRITE_SIZE * (anim_time % SPLASH_SPRITESHEET_WIDTH),
                    SPLASH_SPRITE_SIZE * (anim_time / SPLASH_SPRITESHEET_WIDTH), SPLASH_SPRITE_SIZE, SPLASH_SPRITE_SIZE,
                    0xff);
    if (anim_time++ == ANIMATION_DURATION_SPLASH + ANIMATION_DURATION_SPLASH_EXTRA) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
  else if (anim_state == ANIMATION_HIT) {
    if (anim_time < ANIMATION_DURATION_HIT)
      tigrBlitAlpha(game, explosion, anim_pos.x, anim_pos.y,
                    EXPLOSION_SPRITE_SIZE * (anim_time % EXPLOSION_SPRITESHEET_WIDTH),
                    EXPLOSION_SPRITE_SIZE * (anim_time / EXPLOSION_SPRITESHEET_WIDTH), EXPLOSION_SPRITE_SIZE,
                    EXPLOSION_SPRITE_SIZE, 0xff);
    if (anim_time++ == ANIMATION_DURATION_HIT + ANIMATION_DURATION_HIT_EXTRA) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
}
void updateGameOverlays(Tigr* game, BoardData* p1, BoardData* p2) {
  static int frame = 0;

  int offset_x = TEXT_SPACE;
  for (int i = p1->hit_squares_head; i >= 0; --i) {
    int sx =
        (((frame + fire_frame_offsets[i]) % ANIMATION_DURATION_FLAME) % FLAME_SPRITESHEET_WIDTH) * FLAME_SPRITE_SIZE;
    int sy =
        (((frame + fire_frame_offsets[i]) % ANIMATION_DURATION_FLAME) / FLAME_SPRITESHEET_WIDTH) * FLAME_SPRITE_SIZE;
    tigrBlitAlpha(game, fire_anim, offset_x + (p1->hit_squares[i].x * BOARD_SQUARE_SIZE) - 22,
                  TEXT_SPACE + (p1->hit_squares[i].y * BOARD_SQUARE_SIZE) - 14, sx, sy, FLAME_SPRITE_SIZE,
                  FLAME_SPRITE_SIZE, 0xff);
  }
  offset_x = TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + TEXT_SPACE;
  for (int i = p2->hit_squares_head; i >= 0; --i) {
    int sx = (((frame + fire_frame_offsets[18 + i]) % ANIMATION_DURATION_FLAME) % FLAME_SPRITESHEET_WIDTH) *
             FLAME_SPRITE_SIZE;
    int sy = (((frame + fire_frame_offsets[18 + i]) % ANIMATION_DURATION_FLAME) / FLAME_SPRITESHEET_WIDTH) *
             FLAME_SPRITE_SIZE;
    tigrBlitAlpha(game, fire_anim, offset_x + (p2->hit_squares[i].x * BOARD_SQUARE_SIZE) - 22,
                  TEXT_SPACE + (p2->hit_squares[i].y * BOARD_SQUARE_SIZE) - 14, sx, sy, FLAME_SPRITE_SIZE,
                  FLAME_SPRITE_SIZE, 0xff);
  }
  frame = (frame + 1) % ANIMATION_DURATION_FLAME;

  if (anim_state != ANIMATION_NONE)
    updateGameAnimations(game);
}
void renderLabels(Tigr* game) {
  char s[BUFFER];
  int x, y;
  int x_offset = BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE;
  for (int i = 0; i < BOARD_SIZE; ++i) {
    // Row labels
    sprintf(s, "%d", i + 1);
    x = tigrTextWidth(old_london_25, s), y = tigrTextHeight(old_london_25, s);
    tigrPrint(game, old_london_25, (TEXT_SPACE - x) / 2,
              TEXT_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - y) / 2, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, old_london_25, x_offset - (TEXT_SPACE + x) / 2,
              TEXT_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - y) / 2, (TPixel){0xff, 0xff, 0xff, 0xff}, s);

    // Column labels
    sprintf(s, "%c", i + 65);
    x = tigrTextWidth(old_london_25, s), y = tigrTextHeight(old_london_25, s);
    tigrPrint(game, old_london_25, TEXT_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - x) / 2,
              (TEXT_SPACE - y) / 2, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, old_london_25, x_offset + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - x) / 2, (TEXT_SPACE - y) / 2,
              (TPixel){0xff, 0xff, 0xff, 0xff}, s);
  }
}
void renderIcons(Tigr* game) {
  tigrBlitTint(game, boat_icons, TEXT_SPACE, TEXT_SPACE + BOARD_RENDER_SIZE + 20, 0, 0, boat_icons->w, boat_icons->h,
               SHIP_LIVE_COLOUR);
  tigrBlitTint(game, boat_icons, TEXT_SPACE + BOARD_RENDER_SIZE + TEXT_SPACE + BOARD_SQUARE_SIZE,
               TEXT_SPACE + BOARD_RENDER_SIZE + 20, 0, 0, boat_icons->w, boat_icons->h, SHIP_LIVE_COLOUR);
}
void renderDeath(Tigr* game, int i) {
  int x_offset = i < 5 ? TEXT_SPACE : 2 * TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE, x_sheet = 0;
  for (int j = i / 5; j < i % 5; ++j) {
    x_offset += IconWidths[j];
    x_sheet += IconWidths[j];
  }
  tigrBlitTint(game, boat_icons, x_offset, TEXT_SPACE + BOARD_RENDER_SIZE + 20, x_sheet, 0, IconWidths[i % 5],
               boat_icons->h, SHIP_DEAD_COLOUR);
}
#endif
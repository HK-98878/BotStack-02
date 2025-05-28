// Included first due to the python header needing to be first
#include "include/_globals.h"

#if defined(SOUND) && defined(_WIN32)
#include <windows.h>
#pragma comment(lib, "winmm.lib")  // This adds the static library "winmm.lib" to the project

const char explosions[4][50] = {
    "resources/sounds/explosion_1.wav", "resources/sounds/explosion_2.wav",
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

// Player header
#include "public/player.h"

#define ERROR_MSG(msg)      \
  {                         \
    printf("[ERROR] " msg); \
    exit(1);                \
  }

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
TigrResize* window;

TigrFont *old_london_25, *old_london_50;
Tigr *explosion, *splash, *target, *miss, *fire_anim, *boat_icons, *start_screen, *end_screen;
Tigr *water_spritesheet_blue, *water_spritesheet_red, *water_spritesheet_green;
int backgroundState = 0;

Tigr** boat_spritesheets;
const int BoatSpritesheetSizes[5] = {PATROL_BOAT_SPRITE_SIZE, SUBMARINE_SPRITE_SIZE,
                                     DESTROYER_SPRITE_SIZE, BATTLESHIP_SPRITE_SIZE,
                                     CARRIER_SPRITE_SIZE};
const int IconWidths[5] = {ICON_PATROL_BOAT_WIDTH, ICON_SUBMARINE_WIDTH, ICON_DESTROYER_WIDTH,
                           ICON_BATTLESHIP_WIDTH, ICON_CARRIER_WIDTH};
int fire_frame_offsets[36];

int shootAnim(Tigr* game, BoardData* b, int player1, int x, int y);

void renderBackground(Tigr* game, Tigr* background);
void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2);
void updateGameOverlays(Tigr* game, BoardData* p1, BoardData* p2);
void renderLabels(Tigr* game, const char* player1_name, const char* player2_name);
void renderIcons(Tigr* game);
void renderDeath(Tigr* game, int i);

#define LOAD_IMAGE(img, path)            \
  img = tigrLoadImage(path);             \
  if (img == NULL) {                     \
    printf("Failed to load [" path "]"); \
    return 0;                            \
  }
#endif

#define CHECK_END_GAME(p, player) \
  if (p->remaining_ships == 0) {  \
    flags |= GAME_OVER;           \
  }

static Coordinate toRender(int x, int y, int board) {
  return (Coordinate){
      TEXT_SPACE + (x * BOARD_SQUARE_SIZE) +
          (board == 2) * (BOARD_RENDER_SIZE + TEXT_SPACE + BOARD_SQUARE_SIZE),
      TEXT_SPACE + HEADER_SPACE + (y * BOARD_SQUARE_SIZE),
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

char* exec_str;

#define CMD_BUFFER_LEN   50
#define RESET_CMD_BUFFER memset(cmd_buffer, 0, CMD_BUFFER_LEN), cmd_buffer_ptr = 0;

typedef ShipPlacement (*PlaceFnc)(Board, ShipID);
typedef Coordinate (*ShootFnc)(Board);
typedef struct PlayerDefinition {
  char name[50];

  int usingPython;

  PlaceFnc PlaceFuncPtr;
  ShootFnc ActionFuncPtr;

  char python_name[50];
  PyObject* PlaceFuncPtrPy;
  PyObject* ActionFuncPtrPy;

  int wins;
  int losses;

  int hits;
  int misses;
} PlayerDefinition;
PlayerDefinition* newCPlayer(const char* name, PlaceFnc PlaceFuncPtr, ShootFnc ActionFuncPtr) {
  PlayerDefinition* p = (PlayerDefinition*)calloc(sizeof(PlayerDefinition), 1);
  strcpy(p->name, name);

  p->usingPython = 0;
  p->PlaceFuncPtr = PlaceFuncPtr;
  p->ActionFuncPtr = ActionFuncPtr;

  p->wins = 0;
  p->losses = 0;
  p->hits = 0;
  p->misses = 0;

  return p;
}
PlayerDefinition* newPythonPlayer(const char* name, const char* python_file) {
  PlayerDefinition* p = (PlayerDefinition*)calloc(sizeof(PlayerDefinition), 1);
  strcpy(p->name, name);

  p->usingPython = 1;
  strcpy(p->python_name, python_file);

  p->wins = 0;
  p->losses = 0;
  p->hits = 0;
  p->misses = 0;

  return p;
}
#define PLAYER_COUNT 9

int runGame(PlayerDefinition* player_def_1, PlayerDefinition* player_def_2, PyObject* main_dict) {
  PyObject *main_dict_copy1, *main_dict_copy2;
  if (player_def_1->usingPython) {
    if (!Py_IsInitialized()) {
      printf("Python not initialised");
      exit(1);
    }

    FILE* f = fopen(player_def_1->python_name, "r");
    main_dict_copy1 = PyDict_Copy(main_dict);

    PyRun_File(f, player_def_1->python_name, Py_file_input, main_dict_copy1, main_dict_copy1);
    PyRun_String(exec_str, Py_file_input, main_dict_copy1, main_dict_copy1);

    player_def_1->PlaceFuncPtrPy = PyDict_GetItemString(main_dict_copy1, "player1PlaceShip");
    player_def_1->ActionFuncPtrPy = PyDict_GetItemString(main_dict_copy1, "player1Action");
    if (player_def_1->PlaceFuncPtrPy == NULL || player_def_1->ActionFuncPtrPy == NULL) {
      printf("Error loading python functions for player [%s]", player_def_1->name);
      exit(1);
    }
    printf("Got Python functions\n");

    fclose(f);
  }
  if (player_def_2->usingPython) {
    if (!Py_IsInitialized()) {
      printf("Python not initialised");
      exit(1);
    }

    FILE* f = fopen(player_def_2->python_name, "r");
    main_dict_copy2 = PyDict_Copy(main_dict);

    PyRun_File(f, player_def_2->python_name, Py_file_input, main_dict_copy2, main_dict_copy2);
    PyRun_String(exec_str, Py_file_input, main_dict_copy2, main_dict_copy2);

    player_def_2->PlaceFuncPtrPy = PyDict_GetItemString(main_dict_copy2, "player1PlaceShip");
    player_def_2->ActionFuncPtrPy = PyDict_GetItemString(main_dict_copy2, "player1Action");
    if (player_def_2->PlaceFuncPtrPy == NULL || player_def_2->ActionFuncPtrPy == NULL) {
      printf("Error loading python functions for player [%s]", player_def_2->name);
      exit(1);
    }
    printf("Got Python functions\n");

    fclose(f);
  }

  // ---------- Game variables ----------
  // Gamestate
  unsigned char flags = PLAYER_1;

  // ---------- Player setup ----------
#if defined(NO_GRAPHICS) || defined(LOG)
  fprintf(LOG_LOCATION, "----------\n GAME START\n----------\n\n");
#endif

  tigrBlit(window->contents, start_screen, (window->contents->w - start_screen->w) / 2, 0, 0, 0,
           start_screen->w, start_screen->h);
  tigrPrint(window->contents, old_london_50, 400, 950, (TPixel){0xff, 0xff, 0xff, 0xff},
            player_def_1->name);
  tigrPrint(window->contents, old_london_50,
            1620 - tigrTextWidth(old_london_50, player_def_2->name) / 2, 850,
            (TPixel){0xff, 0xff, 0xff, 0xff}, player_def_2->name);
  int c = 0;
  while (c != '\n') {
    c = tigrReadChar(window->window);

    if (tigrClosed(window->window) || c == 'x') {
      if (player_def_1->usingPython) {
        Py_DECREF(player_def_1->PlaceFuncPtrPy), player_def_1->PlaceFuncPtrPy = NULL;
        Py_DECREF(player_def_1->ActionFuncPtrPy), player_def_1->ActionFuncPtrPy = NULL;
      }
      if (player_def_2->usingPython) {
        Py_DECREF(player_def_2->PlaceFuncPtrPy), player_def_2->PlaceFuncPtrPy = NULL;
        Py_DECREF(player_def_2->ActionFuncPtrPy), player_def_2->ActionFuncPtrPy = NULL;
      }
      tigrClear(window->contents, BG_COLOUR);
      tigrResizeUpdate(window);
      return 1;
    }

    tigrResizeUpdate(window);
  }

  tigrClear(window->contents, BG_COLOUR);
  renderLabels(window->contents, player_def_1->name, player_def_2->name);
  renderIcons(window->contents);

  BoardData* p1 = initBoardData(0);
  BoardData* p2 = initBoardData(1);

  ShipPlacement placement;
  Board* b;
  Coordinate shot;
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p1, 0);
    if (player_def_1->usingPython)
      placement = playerPlaceShipPython(*b, i, player_def_1->PlaceFuncPtrPy);
    else
      placement = (*player_def_1).PlaceFuncPtr(*b, i);

    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR_MSG("Player 1 - Undefined ship placement function")
    }
    if (!placeShip(p1, i, placement.x, placement.y, placement.rotation)) {
      ERROR_MSG("Player 1 - Invalid ship placement")
    }
#if defined(NO_GRAPHICS) || defined(LOG)
    else
      fprintf(LOG_LOCATION, "[Player 1] Place %d | %d %d - %d\n", i, placement.x, placement.y,
              placement.rotation);
#endif
  }
  for (int i = SHIP_PATROL_BOAT; i <= SHIP_CARRIER; ++i) {
    b = toBoard(p2, 0);
    if (player_def_2->usingPython)
      placement = playerPlaceShipPython(*b, i, player_def_2->PlaceFuncPtrPy);
    else
      placement = (*player_def_2).PlaceFuncPtr(*b, i);

    free(b);
    if (placement.x == -1 || placement.y == -1 || placement.rotation == -1) {
      ERROR_MSG("Player 2 - Undefined ship placement function")
    }
    if (!placeShip(p2, i, placement.x, placement.y, placement.rotation)) {
      ERROR_MSG("Player 2 - Invalid ship placement")
    }
#if defined(NO_GRAPHICS) || defined(LOG)
    else
      fprintf(LOG_LOCATION, "[Player 2] Place %d | %d %d - %d\n", i, placement.x, placement.y,
              placement.rotation);
#endif
  }
  flags |= SHIPS_PLACED;

  clock_t t, last_t = clock();
#ifndef NO_GRAPHICS
  while (tigrReadChar(window->window) != 0)
    tigrResizeUpdate(window);
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

    if (tigrReadChar(window->window) == 'x')
      break;

    if (anim_state == ANIMATION_NONE) {
      if (flags & PLAYER_1) {
        b = toBoard(p2, 1);
        printf("[Player 1] Shoot   ");
        if (player_def_1->usingPython)
          shot = playerActionPython(*b, player_def_1->ActionFuncPtrPy);
        else
          shot = (*player_def_1->ActionFuncPtr)(*b);
        free(b), b = NULL;
        if (shot.x == -1 && shot.y == -1) {
          ERROR_MSG("Player 1 - Unimplemented shot function")
        }
#if defined(NO_GRAPHICS)
        fprintf(LOG_LOCATION, "| %d %d\n", shot.x, shot.y);

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
        b = toBoard(p1, 1);
        printf("[Player 2] Shoot   ");
        if (player_def_2->usingPython)
          shot = playerActionPython(*b, player_def_2->ActionFuncPtrPy);
        else
          shot = (*player_def_2->ActionFuncPtr)(*b);
        free(b), b = NULL;
        if (shot.x == -1 && shot.y == -1) {
          ERROR_MSG("Player 2 - Unimplemented shot function")
        }
#ifdef NO_GRAPHICS
        fprintf(LOG_LOCATION, "| %d %d\n", shot.x, shot.y);
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
    renderBackground(window->contents, *bg);
    updateGameRender(window->contents, p1, p2);
    updateGameOverlays(window->contents, p1, p2);
    if (anim_state == ANIMATION_NONE && flags & SHOOT) {
      flags &= ~SHOOT;
      if (flags & PLAYER_1) {
#ifdef LOG
        fprintf(LOG_LOCATION, "| %d %d\n", shot.x, shot.y);
#endif
        if (!shootAnim(window->contents, p2, 0, shot.x, shot.y))
          flags &= ~PLAYER_1;
        CHECK_END_GAME(p2, "1")
      }
      else {
#ifdef LOG
        fprintf(LOG_LOCATION, "| %d %d\n", shot.x, shot.y);
#endif
        if (!shootAnim(window->contents, p1, 1, shot.x, shot.y))
          flags |= PLAYER_1;

        CHECK_END_GAME(p1, "2")
      }
    }
    tigrResizeUpdate(window);
    if (anim_state == ANIMATION_NONE && flags & GAME_OVER) {
      if (flags & PLAYER_1) {
        player_def_1->wins++;
        player_def_2->losses++;
      }
      else {
        player_def_2->wins++;
        player_def_1->losses++;
      }
#ifdef LOG
      fprintf(LOG_LOCATION, "Player %s wins\n", flags & PLAYER_1 ? "1" : "2");
#endif
      break;
    }
#else
    if (flags & GAME_OVER) {
      if (flags & PLAYER_1) {
        player_def_1->wins++;
        player_def_2->losses++;
      }
      else {
        player_def_2->wins++;
        player_def_1->losses++;
      }
      fprintf(LOG_LOCATION, "Player %s wins\n", flags & PLAYER_1 ? "1" : "2");
      break;
    }
#endif
  }

  // Game over or exiting

  // Deassign python objects
  if (player_def_1->usingPython) {
    Py_DECREF(player_def_1->PlaceFuncPtrPy), player_def_1->PlaceFuncPtrPy = NULL;
    Py_DECREF(player_def_1->ActionFuncPtrPy), player_def_1->ActionFuncPtrPy = NULL;
    Py_DECREF(main_dict_copy1), main_dict_copy1 = NULL;
  }
  if (player_def_2->usingPython) {
    Py_DECREF(player_def_2->PlaceFuncPtrPy), player_def_2->PlaceFuncPtrPy = NULL;
    Py_DECREF(player_def_2->ActionFuncPtrPy), player_def_2->ActionFuncPtrPy = NULL;
    Py_DECREF(main_dict_copy2), main_dict_copy2 = NULL;
  }

  // If the game ended properly: end-screen, then fade out
  if (flags & GAME_OVER) {
    tigrClear(window->contents, BG_COLOUR);
    tigrBlit(window->contents, end_screen, (window->contents->w - end_screen->w) / 2, 0, 0, 0,
             end_screen->w, end_screen->h);
    tigrPrint(window->contents, old_london_50, 415, 50, (TPixel){0xff, 0xff, 0xff, 0xff},
              (flags & PLAYER_1) ? player_def_2->name : player_def_1->name);

    int i = 20;
    while (i > 0) {
      t = clock();
      if (t - last_t < FRAME_TIME)
        continue;
      last_t = t;

      tigrResizeNoUpdate(window);
      tigrFillRect(window->window, 0, 0, window->window->w, window->window->h,
                   (TPixel){0x00, 0x00, 0x00, i-- * 0xff / 20});
      tigrUpdate(window->window);
    }

    while (tigrReadChar(window->window) != '\n') {
      if (tigrClosed(window->window))
        return 1;
      tigrResizeUpdate(window);
    }

    while (i < 20) {
      t = clock();
      if (t - last_t < FRAME_TIME)
        continue;
      last_t = t;

      tigrResizeNoUpdate(window);
      tigrFillRect(window->window, 0, 0, window->window->w, window->window->h,
                   (TPixel){0x00, 0x00, 0x00, ++i * 0xff / 20});
      tigrUpdate(window->window);
    }
  }

  tigrClear(window->contents, BG_COLOUR);
  tigrResizeUpdate(window);

  return 0;
}

int main() {
  char cmd_buffer[CMD_BUFFER_LEN];
  int cmd_buffer_ptr = 0;

#ifndef NO_GRAPHICS
  // ---------- Window setup ----------
  Tigr* game = tigrBitmap(2 * BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                          TEXT_SPACE + HEADER_SPACE + BOARD_RENDER_SIZE + ICON_SPACE);
  tigrClear(game, BG_COLOUR);
  game->blitMode = TIGR_KEEP_ALPHA;

  // ---------- Load resources ----------
  Tigr* old_london_25_img = tigrLoadImage("resources/OldLondon_25.png");
  if (old_london_25_img == NULL) {
    ERROR_MSG("Failed to load font image")
  }
  old_london_25 = tigrLoadFont(old_london_25_img, TCP_1252);
  if (old_london_25 == NULL) {
    ERROR_MSG("Failed to load font")
  }
  Tigr* old_london_50_img = tigrLoadImage("resources/OldLondon_50.png");
  if (old_london_50_img == NULL) {
    ERROR_MSG("Failed to load font image")
  }
  old_london_50 = tigrLoadFont(old_london_50_img, TCP_1252);
  if (old_london_50 == NULL) {
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
  LOAD_IMAGE(start_screen, "resources/StartScreen.png");
  LOAD_IMAGE(end_screen, "resources/EndScreen.png");

  boat_spritesheets = calloc(5, sizeof(Tigr*));
  LOAD_IMAGE(boat_spritesheets[0], "resources/_Spritesheet_PatrolBoat.png")
  LOAD_IMAGE(boat_spritesheets[1], "resources/_Spritesheet_Submarine.png")
  LOAD_IMAGE(boat_spritesheets[2], "resources/_Spritesheet_Destroyer.png")
  LOAD_IMAGE(boat_spritesheets[3], "resources/_Spritesheet_Battleship.png")
  LOAD_IMAGE(boat_spritesheets[4], "resources/_Spritesheet_Carrier.png")

  for (int i = 0; i < 36; ++i) {
    fire_frame_offsets[i] = rand() % ANIMATION_DURATION_FLAME;
  }

  window = (TigrResize*)calloc(1, sizeof(TigrResize));
  window->window = tigrWindow(game->w, game->h, "Game", TIGR_AUTO);
  window->contents = game;
#endif

  Py_Initialize();
  PyObject* main_module = PyImport_AddModule("__main__");
  PyObject* main_dict = PyModule_GetDict(main_module);

  exec_str = calloc(500, sizeof(char));
  sprintf(
      exec_str,
      "BOARD_SIZE=%d\nSHIP_PATROL_BOAT=%d\nSHIP_SUBMARINE=%d\nSHIP_DESTROYER=%d\nSHIP_BATTLESHIP="
      "%d\nSHIP_CARRIER=%d\nSQUARE_EMPTY=%d\nSQUARE_MISS=%d\nSQUARE_HIT=%d\nSQUARE_SHIP_PATROL_"
      "BOAT="
      "%d\nSQUARE_SHIP_SUBMARINE=%d\nSQUARE_SHIP_DESTROYER=%d\nSQUARE_SHIP_BATTLESHIP="
      "%d\nSQUARE_SHIP_CARRIER=%d\nSHOT_MISS=%d\nSHOT_HIT=%d\nSHOT_SUNK=%d\n",
      BOARD_SIZE, SHIP_PATROL_BOAT, SHIP_SUBMARINE, SHIP_DESTROYER, SHIP_BATTLESHIP, SHIP_CARRIER,
      SQUARE_EMPTY, SQUARE_MISS, SQUARE_HIT, SQUARE_SHIP_PATROL_BOAT, SQUARE_SHIP_SUBMARINE,
      SQUARE_SHIP_DESTROYER, SQUARE_SHIP_BATTLESHIP, SQUARE_SHIP_CARRIER, SHOT_MISS, SHOT_HIT,
      SHOT_HIT_SUNK);

#if defined(LOG) && LOG != 1
  f = fopen(LOG, "ab+");
#endif

  srand(time(0));

  // PlayerDefinition* players[PLAYER_COUNT] = {newCPlayer("Player 1", &player1PlaceShip,
  // &player1Action),
  //                                            newCPlayer("Player 2", &player1PlaceShip,
  //                                            &player1Action), newPythonPlayer("Player 3",
  //                                            "testplayers/player1.py")};
  PlayerDefinition* players[PLAYER_COUNT] = {
      newCPlayer("Ethan", &ethanPlaceShip, &ethanAction),
      newCPlayer("Jeson", &jesonPlaceShip, &jesonAction),
      newCPlayer("Tom", &tomPlaceShip, &tomAction),
      newCPlayer("Vin & Artemis", &vinPlaceShip, &vinAction),
      newCPlayer("Yash", &yashPlaceShip, &yashAction),
      newPythonPlayer("Harrish", "players/pyplayer_harrish.py"),
      newPythonPlayer("Mehul", "players/pyplayer_mehul.py"),
      newPythonPlayer("Helitha", "players/pyplayer_savira.py"),
      newPythonPlayer("Tara", "players/pyplayer_tara.py"),
  };

  int group[5] = {-1, -1, -1, -1, -1};

  RESET_CMD_BUFFER
#ifndef NO_GRAPHICS
  int run = 1;
  while (run) {
    printf("Enter Command:\n");
    while (1) {
      if (tigrClosed(window->window)) {
        RESET_CMD_BUFFER
        run = 0;
        break;
      }
      tigrResizeUpdate(window);
      int c = tigrReadChar(window->window);
      if (c == 0)
        continue;
      if (c == '\n') {
        break;
      }

      printf("%c", c);
      cmd_buffer[cmd_buffer_ptr++] = (char)c;
      if (cmd_buffer_ptr == 50) {
        printf("\n Command Overflow\n");
        RESET_CMD_BUFFER
      }
    }
    printf("\n");
    // Process command
    int index1, index2;
    switch (cmd_buffer[0]) {
      case 'r':
        printf("Confirm reset all scores: ");
        while (1) {
          tigrResizeUpdate(window);
          int c = tigrReadChar(window->window);
          if (c == 0)
            continue;
          if (c == 'y') {
            for (int i = 0; i < PLAYER_COUNT; ++i) {
              players[i]->wins = 0;
              players[i]->losses = 0;
            }
          }
          else
            break;
        }
        printf("\n");
        break;
      case 'g':
        if (cmd_buffer[1] == 'c') {
          printf("Clearing Group\n");
          for (int i = 0; i < 5; ++i) {
            group[i] = -1;
          }
        }
        else if (cmd_buffer[1] >= '0' && cmd_buffer[1] <= '4' && cmd_buffer[2] >= '0' &&
                 cmd_buffer[2] <= '0' + PLAYER_COUNT - 1) {
          printf("Adding [%s] to the group\n", players[cmd_buffer[2] - '0']->name);
          group[cmd_buffer[1] - '0'] = cmd_buffer[2] - '0';
        }
        break;
      case 'l':
        if (cmd_buffer[1] == 'f') {
          printf("Listing all members:\n");
          for (int i = 0; i < PLAYER_COUNT; ++i) {
            printf("%d | %s \t| %d %d\n", i, players[i]->name, players[i]->wins,
                   players[i]->losses);
          }
        }
        else {
          printf("Listing group members:\n");
          for (int i = 0; i < 5; ++i) {
            if (group[i] != -1)
              printf("%s \t| %d %d\n", players[group[i]]->name, players[group[i]]->wins,
                     players[group[i]]->losses);
          }
        }
        break;
      case 'm':
        index1 = cmd_buffer[1] - '0';
        index2 = cmd_buffer[2] - '0';
        if (index1 < 0 || index1 > PLAYER_COUNT - 1 || index2 < 0 || index2 > PLAYER_COUNT - 1)
          break;
        printf("Running match between %s & %s\n", players[index1]->name, players[index2]->name);
#if defined(LOG) && LOG != 1
        fprintf(LOG_LOCATION, "Running match between %s & %s\n", players[index1]->name,
                players[index2]->name);
#endif
        runGame(players[index1], players[index2], main_dict);
        break;
      case 'b':
        int bg = cmd_buffer[1] - '0';
        if (bg >= 0 && bg <= 2) {
          printf("Set Background State - %d\n", bg);
          backgroundState = bg;
        }
        break;
      case 's':
        if (cmd_buffer[2] != 'w' && cmd_buffer[2] != 'l')
          break;
        index1 = cmd_buffer[1] - '0';
        int val = cmd_buffer[3] - '0';
        if (index1 < 0 || index1 > PLAYER_COUNT - 1 || val < 0 || val > 9)
          break;
        if (cmd_buffer[2] == 'w') {
          printf("Setting wins for player [%s] to %d", players[index1]->name, val);
          players[index1]->wins = val;
        }
        else if (cmd_buffer[2] == 'w') {
          printf("Setting losses for player [%s] to %d", players[index1]->name, val);
          players[index1]->losses = val;
        }
        break;
      case 'q':
        printf("Confirm quit: ");
        while (1) {
          tigrResizeUpdate(window);
          int c = tigrReadChar(window->window);
          if (c == 0)
            continue;
          if (c == 'y') {
            for (int i = 0; i < PLAYER_COUNT; ++i) {
              players[i]->wins = 0;
              players[i]->losses = 0;
            }
          }
          else
            break;
        }
        run = 0;
        break;
      default:
        printf("Invalid command\n");
        break;
    }
    RESET_CMD_BUFFER
  }
#endif

  // End

  free(exec_str), exec_str = NULL;

#ifndef NO_GRAPHICS
  tigrFree(water_spritesheet_blue), tigrFree(water_spritesheet_green),
      tigrFree(water_spritesheet_red);
  tigrFreeFont(old_london_25), tigrFreeFont(old_london_50);

  tigrFree(game);
  tigrFree(window->window);
  if (window->contents_display != NULL)
    tigrFree(window->contents_display);
  free(window);
#endif

  Py_DECREF(main_dict), main_dict = NULL;
  Py_Finalize();

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

  tigrBlit(game, background, TEXT_SPACE, TEXT_SPACE + HEADER_SPACE,
           frame_offset + (frame * WATER_SPRITE_SIZE_X), 0, BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);
  tigrBlit(game, background, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
           TEXT_SPACE + HEADER_SPACE, frame_offset + (frame)*WATER_SPRITE_SIZE_X, 0,
           BOARD_RENDER_SIZE, BOARD_RENDER_SIZE);

  frame = (frame + 1) % ANIMATION_DURATION_WATER;
  if (frame_offset == WATER_SPRITE_SIZE_X - BOARD_RENDER_SIZE)
    frame_offset_dir = -1;
  else if (frame_offset == 0)
    frame_offset_dir = 1;
  frame_offset = (frame_offset + (2 * frame_offset_dir));
}

void updateGameRender(Tigr* game, BoardData* p1, BoardData* p2) {
  tigrBlitAlpha(game, p1->board_render, TEXT_SPACE, TEXT_SPACE + HEADER_SPACE, 0, 0,
                BOARD_RENDER_SIZE, BOARD_RENDER_SIZE, 0xff);
  tigrBlitAlpha(game, p2->board_render, BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE,
                TEXT_SPACE + HEADER_SPACE, 0, 0, BOARD_RENDER_SIZE, BOARD_RENDER_SIZE, 0xff);
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
      tigrBlitAlpha(game, target, anim_pos.x, anim_pos.y, 0, 0, BOARD_SQUARE_SIZE,
                    BOARD_SQUARE_SIZE, 0xff);
      tigrFillCircle(game, anim_pos.x + BOARD_SQUARE_SIZE / 2, anim_pos.y + BOARD_SQUARE_SIZE / 2,
                     r, (TPixel){0x00, 0x00, 0x00, 0xff});
    }
    if (anim_time++ == ANIMATION_DURATION_SHOOT) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
  else if (anim_state == ANIMATION_SPLASH) {
    if (anim_time < ANIMATION_DURATION_SPLASH)
      tigrBlitAlpha(game, splash, anim_pos.x, anim_pos.y,
                    SPLASH_SPRITE_SIZE * (anim_time % SPLASH_SPRITESHEET_WIDTH),
                    SPLASH_SPRITE_SIZE * (anim_time / SPLASH_SPRITESHEET_WIDTH), SPLASH_SPRITE_SIZE,
                    SPLASH_SPRITE_SIZE, 0xff);
    if (anim_time++ == ANIMATION_DURATION_SPLASH + ANIMATION_DURATION_SPLASH_EXTRA) {
      anim_time = 0;
      anim_state = ANIMATION_NONE;
    }
  }
  else if (anim_state == ANIMATION_HIT) {
    if (anim_time < ANIMATION_DURATION_HIT)
      tigrBlitAlpha(game, explosion, anim_pos.x, anim_pos.y,
                    EXPLOSION_SPRITE_SIZE * (anim_time % EXPLOSION_SPRITESHEET_WIDTH),
                    EXPLOSION_SPRITE_SIZE * (anim_time / EXPLOSION_SPRITESHEET_WIDTH),
                    EXPLOSION_SPRITE_SIZE, EXPLOSION_SPRITE_SIZE, 0xff);
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
        (((frame + fire_frame_offsets[i]) % ANIMATION_DURATION_FLAME) % FLAME_SPRITESHEET_WIDTH) *
        FLAME_SPRITE_SIZE;
    int sy =
        (((frame + fire_frame_offsets[i]) % ANIMATION_DURATION_FLAME) / FLAME_SPRITESHEET_WIDTH) *
        FLAME_SPRITE_SIZE;
    tigrBlitAlpha(game, fire_anim, offset_x + (p1->hit_squares[i].x * BOARD_SQUARE_SIZE) - 22,
                  TEXT_SPACE + HEADER_SPACE + (p1->hit_squares[i].y * BOARD_SQUARE_SIZE) - 14, sx,
                  sy, FLAME_SPRITE_SIZE, FLAME_SPRITE_SIZE, 0xff);
  }
  offset_x = TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + TEXT_SPACE;
  for (int i = p2->hit_squares_head; i >= 0; --i) {
    int sx = (((frame + fire_frame_offsets[18 + i]) % ANIMATION_DURATION_FLAME) %
              FLAME_SPRITESHEET_WIDTH) *
             FLAME_SPRITE_SIZE;
    int sy = (((frame + fire_frame_offsets[18 + i]) % ANIMATION_DURATION_FLAME) /
              FLAME_SPRITESHEET_WIDTH) *
             FLAME_SPRITE_SIZE;
    tigrBlitAlpha(game, fire_anim, offset_x + (p2->hit_squares[i].x * BOARD_SQUARE_SIZE) - 22,
                  TEXT_SPACE + HEADER_SPACE + (p2->hit_squares[i].y * BOARD_SQUARE_SIZE) - 14, sx,
                  sy, FLAME_SPRITE_SIZE, FLAME_SPRITE_SIZE, 0xff);
  }
  frame = (frame + 1) % ANIMATION_DURATION_FLAME;

  if (anim_state != ANIMATION_NONE)
    updateGameAnimations(game);
}
void renderLabels(Tigr* game, const char* player1_name, const char* player2_name) {
  char s[BUFFER];
  int x, y;
  int x_offset = BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + 2 * TEXT_SPACE;
  for (int i = 0; i < BOARD_SIZE; ++i) {
    // Row labels
    sprintf(s, "%d", i + 1);
    x = tigrTextWidth(old_london_25, s), y = tigrTextHeight(old_london_25, s);
    tigrPrint(game, old_london_25, (TEXT_SPACE - x) / 2,
              TEXT_SPACE + HEADER_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - y) / 2,
              (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, old_london_25, x_offset - (TEXT_SPACE + x) / 2,
              TEXT_SPACE + HEADER_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - y) / 2,
              (TPixel){0xff, 0xff, 0xff, 0xff}, s);

    // Column labels
    sprintf(s, "%c", i + 65);
    x = tigrTextWidth(old_london_25, s), y = tigrTextHeight(old_london_25, s);
    tigrPrint(game, old_london_25, TEXT_SPACE + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - x) / 2,
              (TEXT_SPACE - y) / 2 + HEADER_SPACE, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
    tigrPrint(game, old_london_25, x_offset + i * BOARD_SQUARE_SIZE + (BOARD_SQUARE_SIZE - x) / 2,
              (TEXT_SPACE - y) / 2 + HEADER_SPACE, (TPixel){0xff, 0xff, 0xff, 0xff}, s);
  }

  tigrPrint(game, old_london_50,
            TEXT_SPACE + BOARD_RENDER_SIZE / 2 - tigrTextWidth(old_london_50, player1_name) / 2, 10,
            (TPixel){0xff, 0xff, 0xff, 0xff}, player1_name);
  tigrPrint(game, old_london_50,
            TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE + TEXT_SPACE +
                BOARD_RENDER_SIZE / 2 - tigrTextWidth(old_london_50, player1_name) / 2,
            10, (TPixel){0xff, 0xff, 0xff, 0xff}, player2_name);
}
void renderIcons(Tigr* game) {
  tigrBlitTint(game, boat_icons, TEXT_SPACE, TEXT_SPACE + HEADER_SPACE + BOARD_RENDER_SIZE + 20, 0,
               0, boat_icons->w, boat_icons->h, SHIP_LIVE_COLOUR);
  tigrBlitTint(game, boat_icons, TEXT_SPACE + BOARD_RENDER_SIZE + TEXT_SPACE + BOARD_SQUARE_SIZE,
               TEXT_SPACE + HEADER_SPACE + BOARD_RENDER_SIZE + 20, 0, 0, boat_icons->w,
               boat_icons->h, SHIP_LIVE_COLOUR);
}
void renderDeath(Tigr* game, int i) {
  int x_offset = i < 5 ? TEXT_SPACE : 2 * TEXT_SPACE + BOARD_RENDER_SIZE + BOARD_SQUARE_SIZE,
      x_sheet = 0;
  for (int j = i / 5; j < i % 5; ++j) {
    x_offset += IconWidths[j];
    x_sheet += IconWidths[j];
  }
  tigrBlitTint(game, boat_icons, x_offset, TEXT_SPACE + HEADER_SPACE + BOARD_RENDER_SIZE + 20,
               x_sheet, 0, IconWidths[i % 5], boat_icons->h, SHIP_DEAD_COLOUR);
}
#endif
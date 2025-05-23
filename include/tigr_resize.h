#ifndef TIGR_RESIZE_H
#define TIGR_RESIZE_H

#include "tigr.h"

typedef struct TigrResize {
  Tigr *window;
  Tigr *contents;
  Tigr *contents_display;

  TPixel bg_colour;

  int last_w, last_h;
} TigrResize;

// Scale a provided Tigr bitmap to the size [dx,dy] using bilinear interpolation
Tigr *scaleBitmap(Tigr *src, int dx, int dy);

void tigrResizeUpdate(TigrResize *resize);

#endif
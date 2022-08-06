#include "matrix.h"
#include "fv.h"

Vector
InputAGesture()
{
  static FV fv;

  int x, y; long t; Vector v;

  /*FvAlloc() is typically called only once per program invocation. */
  if(fv == NULL) fv = FvAlloc();
  /* A prototypical loop to compute a feature vector from a gesture being read from a window manager:*/
  FvInit(fv);
  while(GetNextPoint(&x, &y, &t) != END_OF_GESTURE)
    FvAddPoint(fv, x, y, t);
  v = FvCalc(fv);
  return v;
}


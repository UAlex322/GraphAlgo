#ifndef HEAP_H 
#define HEAP_H

#include "graphio.h"

typedef crsGraph crsSpMatrix;
void mspmm_heap(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C);

#endif 
#ifndef spmv_h
#define spmv_h


#include "graphio.h"

typedef crsGraph crsSpMatrix;

void spm_dv_mult(crsSpMatrix *mtx, double *vec, double *res);
void dv_spm_mult(crsSpMatrix *mtx, double *vec, double *res);
int spm_transpose(crsSpMatrix *mtx);

#endif
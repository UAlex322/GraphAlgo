#ifndef MCA_H
#define MCA_H

#include "graphio.h"

struct MCA {
	double *values;
	char   *states;
	size_t  len;
};

typedef crsGraph crsSpMatrix;
void mspmm_mca(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C);

#endif

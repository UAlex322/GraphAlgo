#ifndef MSA_H
#define MSA_H

#include "graphio.h"

struct MSA {
	double *values;
	char   *states;
	size_t  len;
};

typedef crsGraph crsSpMatrix;
void mspmm_msa(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C);

#endif
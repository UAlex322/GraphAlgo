//
//  graphio.h
//  GraphIO
//
//  Created by Andrew Lebedev on 29.09.2020.
//

#ifndef graphio_h
#define graphio_h
#include "mmio.h"

typedef struct  {
    int vertex;
    double val;
} edge;

typedef struct  {
    int* Adjncy;
    int* Xadj;
    double* Eweights;
    MM_typecode matcode;
    int V;
    int nz;
} crsGraph;


int init_graph(crsGraph* gr);
int free_graph_pointers(crsGraph* gr);
int read_mtx_to_crs(crsGraph* gr, const char* filename);
int read_gr_to_crs(crsGraph* gr, const char* filename);
int write_crs_to_mtx(crsGraph* gr, const char* filename);
int read_arr_from_bin(double* arr, int size, const char* filename);
int write_arr_to_bin(double* arr, int size, const char* filename);
int write_arr_to_txt(double* arr, int size, const char* filename);

#endif /* graphio_h */

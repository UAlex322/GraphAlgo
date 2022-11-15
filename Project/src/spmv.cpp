#include "spmv.h"
#include "string.h"
void spm_dv_mult(crsSpMatrix *mtx, double *vec, double *res) {
    for (int i = 0; i < mtx->V; ++i)
        for (int j = mtx->Xadj[i]; j < mtx->Xadj[i+1]; ++j)
            res[i] += mtx->Eweights[j] * vec[mtx->Adjncy[j]];
}

void dv_spm_mult(crsSpMatrix *mtx, double *vec, double *res) {
    for (int i = 0; i < mtx->V; ++i)
        for (int j = mtx->Xadj[i]; j < mtx->Xadj[i+1]; ++j)
            res[mtx->Adjncy[j]] += mtx->Eweights[j] * vec[i];
}

int spm_transpose(crsSpMatrix *mtx) {
    double *newEweights = (double*)malloc((mtx->nz) * sizeof(double)); // new weights array
    int *newAdjncy      = (int*)malloc((mtx->nz) * sizeof(int));       // new adjacency array
    int *colXadj        = (int*)malloc((mtx->V + 1) * sizeof(int));    // column indices
    int *crsCurrPos     = (int*)malloc((mtx->V + 1) * sizeof(int));    // current positions in columns during transposing
    if (!(newEweights && newAdjncy && colXadj && crsCurrPos))
        return -1;

    // filling the column indices array and current column positions array
    memset(colXadj, 0, (mtx->V + 1) * sizeof(int));
    for (int i = 0; i < mtx->nz; ++i)
        ++colXadj[mtx->Adjncy[i]+1];
    colXadj[0] = 0;
    crsCurrPos[0] = 0;
    for (int i = 0; i < mtx->V; ++i) {
        colXadj[i+1] += colXadj[i];
        crsCurrPos[i+1] = colXadj[i+1];
    }

    // transposing
    for (int i = 0; i < mtx->V; ++i) {
        for (int j = mtx->Xadj[i]; j < mtx->Xadj[i+1]; ++j) {
            newEweights[crsCurrPos[mtx->Adjncy[j]]] = mtx->Eweights[j];
            newAdjncy[crsCurrPos[mtx->Adjncy[j]]++] = i;
        }
    }

    // memory deallocation
    free(mtx->Adjncy);
    free(mtx->Xadj);
    free(mtx->Eweights);
    free(crsCurrPos);

    // assigning new data to CRS-matrix
    mtx->Adjncy = newAdjncy;
    mtx->Xadj = colXadj;
    mtx->Eweights = newEweights;

    return 0;
}
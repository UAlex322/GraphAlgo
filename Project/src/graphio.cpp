//
//  graphio.c
//  GraphIO
//
//  Created by Andrew Lebedew on 29.09.2020.
//

#include "graphio.h"

int init_graph(crsGraph* gr) {
    gr -> Adjncy = NULL;
    gr -> Xadj = NULL;
    gr -> Eweights = NULL;
    return 0;
}

int free_graph_pointers(crsGraph* gr) {
    if (!(gr -> Adjncy) && !(gr -> Xadj) && !(gr -> Eweights)) {
        printf("Graph is empty\n");
        return 1;
    }
    if (gr -> Adjncy)
        free(gr -> Adjncy);
    if (gr -> Xadj)
        free(gr -> Xadj);
    if (gr -> Eweights)
        free(gr -> Eweights);
    gr -> Adjncy = NULL;
    gr -> Xadj = NULL;
    gr -> Eweights = NULL;
    return 0;
}

int read_mtx_to_crs(crsGraph* gr, const char* filename) {
    
    /* variables */
    int N, i, row, col, nz_size, curr;
    int *edge_num, *last_el, *row_a, *col_a;
    double val, *val_a;
    fpos_t position;
    FILE *file;
    
    /* mtx correctness check */
    if ((file = fopen(filename, "r")) == NULL) {
        printf("Cannot open file\n");
        return 1;
    }
    if (mm_read_banner(file, &(gr -> matcode))) {
        return 1;
    }
    if (mm_read_mtx_crd_size(file, &(gr -> V), &N, &(gr -> nz))) {
        return 1;
    }
    if (mm_is_complex(gr -> matcode) || mm_is_array(gr -> matcode)) {
        printf("This application doesn't support %s", mm_typecode_to_str(gr -> matcode));
        return 1;
    }
    if (N != (gr -> V)) {
        printf("Is not a square matrix\n");
        return 1;
    }
    
    /* Allocating memmory to store adjacency list */
    last_el = (int*)malloc(sizeof(int) * gr -> V);
    edge_num = (int*)malloc(sizeof(int) * gr -> V);
    if (mm_is_symmetric(gr->matcode)) {
        row_a = (int*)malloc(sizeof(int) * 2 * gr -> nz);
        col_a = (int*)malloc(sizeof(int) * 2 * gr -> nz);
        val_a = (double*)malloc(sizeof(double) * 2 * gr -> nz);
    }
    else {
        row_a = (int *)malloc(sizeof(int) * gr->nz);
        col_a = (int *)malloc(sizeof(int) * gr->nz);
        val_a = (double *)malloc(sizeof(double) * gr->nz);
    }
    for (i = 0; i < (gr -> V); i++) {
        edge_num[i] = 0;
    }
    
    /* Saving value of nz so we can change it */
    nz_size = gr -> nz;

    /* Reading file to count degrees of each vertex */
    curr = 0;
    for(i = 0; i < nz_size; i++) {
       fscanf(file, "%d %d %lg", &row, &col, &val);
       row--;
       col--;
       if (row == col) {
           gr -> nz --;
           continue; //we don't need loops
       }
       row_a[curr] = row;
       col_a[curr] = col;
       val_a[curr++] = val;
       edge_num[row]++;
       if (mm_is_symmetric(gr -> matcode)) {
           edge_num[col]++;
           gr -> nz ++;
           row_a[curr] = col;
           col_a[curr] = row;
           val_a[curr++] = val;
       }
    }

    /* Checking if graph already has arrays */
    if ((gr -> Adjncy != NULL) || (gr -> Xadj != NULL) || (gr -> Eweights != NULL)) {
       free_graph_pointers(gr);
    }

    /* Creating CRS arrays */
    gr -> Adjncy = (int*)malloc(sizeof(int) * (gr -> nz));
    gr -> Xadj = (int*)malloc(sizeof(int) * ((gr -> V) + 1));
    gr -> Eweights = (double*)malloc(sizeof(double) * (gr -> nz));

    /* Writing data in Xadj and last_el */
    gr -> Xadj[0] = 0;
    for(i = 0; i < gr -> V; i++) {
       gr -> Xadj[i+1] = gr -> Xadj[i] + edge_num[i];
       last_el[i] = gr -> Xadj[i];
    }

    /* Reading file to write it's content in crs */
    for(i = 0; i < gr->nz; i++) {
       gr -> Adjncy[last_el[row_a[i]]] = col_a[i];
       gr -> Eweights[last_el[row_a[i]]++] = val_a[i];
    }

    free(edge_num);
    free(last_el);
    free(row_a);
    free(col_a);
    free(val_a);
    fclose(file);
    return 0;
}

int read_gr_to_crs(crsGraph* gr, const char* filename) {
    int i, row, col;
    int *edge_num, *last_el;
    double val;
    char sym = 'c';
    char str[101];
    fpos_t position;
    FILE *file;
    
    /* checking if we can read file */
    if ((file = fopen(filename, "r")) == NULL) {
        printf("Cannot open file\n");
        return 1;
    }

    while (sym == 'c') {
        sym = fgetc(file);
        if (sym == 'p') {
            fscanf(file, "%100s %d %d", str, &gr -> V, &gr -> nz);
            fgets(str, sizeof(str), file);
            fgetpos(file, &position);
        } else {
            fgets(str, sizeof(str), file);
        }
    }

    /* Allocating memmory to store adjacency list */
    last_el = (int*)malloc(sizeof(int) * gr -> V);
    edge_num = (int*)malloc(sizeof(int) * gr -> V);
    
    for (i = 0; i < (gr -> V); i++) {
        edge_num[i] = 0;
    }

    while ((sym = fgetc(file)) != EOF) {
        if (sym == 'a') {
            fscanf(file, "%d %d %lg", &row, &col, &val);
            row--;
            col--;
            if (row == col) {
                gr -> nz --; // We don't need loops
            } else {
                edge_num[row]++;
            }
        }
        fgets(str, sizeof(str), file); // Moving to a new line
    }

    /* Checking if graph already has arrays */
    if ((gr -> Adjncy != NULL) || (gr -> Xadj != NULL) || (gr -> Eweights != NULL)) {
       free_graph_pointers(gr);
    }

    /* Creating CRS arrays */
    gr -> Adjncy = (int*)malloc(sizeof(int) * (gr -> nz));
    gr -> Xadj = (int*)malloc(sizeof(int) * ((gr -> V) + 1));
    gr -> Eweights = (double*)malloc(sizeof(double) * (gr -> nz));

    /* Writing data in Xadj and last_el */
    gr -> Xadj[0] = 0;
    for(i = 0; i < gr -> V; i++) {
       gr -> Xadj[i+1] = gr -> Xadj[i] + edge_num[i];
       last_el[i] = gr -> Xadj[i];
    }

    /* Setting right position */
    fsetpos(file, &position);

    /* Reading file to write it's content in crs */
    while ((sym = fgetc(file)) != EOF) {
        if (sym == 'a'){
            fscanf(file, "%d %d %lg", &row, &col, &val);
            row--;
            col--;
            if (row == col) {
                fgets(str, sizeof(str), file);
                continue; //we don't need loops
            }
            gr -> Adjncy[last_el[row]] = col;
            gr -> Eweights[last_el[row]] = val;
            last_el[row]++;
            fgets(str, sizeof(str), file);
        } else {
            fgets(str, sizeof(str), file);
        }
    }

    free(edge_num);
    free(last_el);
    fclose(file);
    return 0;
}

int write_crs_to_mtx(crsGraph* gr, const char* filename) {
    int i,j;
    FILE* f;
    if ((f = fopen(filename, "w")) == NULL) {
        printf("Can't open file\n");
        return 1;
    }
    
    /* Writing banner and size in mtx */
    mm_write_banner(f, gr -> matcode);
    if(mm_is_symmetric(gr -> matcode)) {
        mm_write_mtx_crd_size(f, gr -> V, gr -> V, gr -> nz/2);
    } else {
        mm_write_mtx_crd_size(f, gr -> V, gr -> V, gr -> nz);
    }
    
    for(i = 0; i < gr -> V; i++) {
        for(j = gr -> Xadj[i]; j < gr -> Xadj[i+1]; j++) {
            if (i > gr -> Adjncy[j] || !mm_is_symmetric(gr -> matcode)) {
                fprintf(f, "%d %d %lg\n", i + 1, gr -> Adjncy[j] + 1, gr -> Eweights[j]);
            }
        }
    }
    fclose(f);
    return 0;
}

int read_arr_from_bin(double* arr, int size, const char* filename) {
    int result;
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Couldn't opem file\n");
        return 1;
    }
    result = fread(arr, sizeof(double), size, file);
    fclose(file);
    if (result == size) {
        return 0;
    } else {
        printf("Reading error\n");
        return 1;
    }
}

int write_arr_to_bin(double* arr, int size, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file){
        printf("Couldn't opem file\n");
        return 1;
    }
    fwrite(arr, sizeof(double), size, file);
    fclose(file);
    return 0;
}

int write_arr_to_txt(double* arr, int size, const char* filename) {
    int i;
    FILE* file = fopen(filename, "w");
    if (!file){
        printf("Couldn't opem file\n");
        return 1;
    }
    for(i = 0; i < size; i++) {
        fprintf(file, "%lg\n", arr[i]);
    }
    fclose(file);
    return 0;
}

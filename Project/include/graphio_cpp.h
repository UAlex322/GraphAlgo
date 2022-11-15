#ifndef graphio_cpp_h
#define graphio_cpp_h
#include "mmio.h"
#include <iostream>
#include <fstream>
#include <string>

template <typename T>
class crsGraphT {
public:
    int* Adj = NULL;
    int* Xdj = NULL;
    T*   Wgt = (T*)NULL;
    MM_typecode matcode;
    int v;
    int nz;

    crsGraphT(const char *filename) {
        if(read_mtx_to_crs(filename))
            throw "Can't read MTX from file";
    }

    crsGraphT() {}

    crsGraphT(const crsGraphT &copy): v(copy.v), nz(copy.nz) {
        memcpy(matcode, mov.matcode, 4);
        Adj = new int[nz];
            memcpy(Adj, copy.Adj, nz*sizeof(int));
        Xdj = new int[v+1];
            memcpy(Xdj, copy.Xdj, (v+1)*sizeof(int));
        if (copy.Wgt != NULL) {
            Wgt = new T[nz];
            for (int i = 0; i < nz; ++i)
                Wgt[i] = copy.Wgt[i];
        }
    }

    crsGraphT(crsGraphT &&mov): v(mov.v), nz(mov.nz) {
        memcpy(matcode, mov.matcode, 4);
        Adj = mov.Adj;
        Xdj = mov.Xdj;
        Wgt = mov.Wgt;
        mov.Adj = NULL;
        mov.Xdj = NULL;
        mov.Wgt = (T*)NULL;
    }

    ~crsGraphT() {
        if (Adj)
            delete[] Adj;
        if (Xdj)
            delete[] Xdj;
        if (Wgt)
            delete[] Wgt;

        Adj = Xdj = NULL;
        Wgt = (T*)NULL;
    }

    void print_crs() {
        cout << gr.V << ' ' << gr.nz << '\n';
        if (gr.Wgt == NULL) {
            for (int i = 0; i < gr.V; ++i)
                for (int j = gr.Xdj[i]; j < gr.Xdj[i+1]; ++j)
                    cout << i+1 << ' ' << gr.Adj[j]+1 << '\n';
        }
        else {
            for (int i = 0; i < gr.V; ++i)
                for (int j = gr.Xdj[i]; j < gr.Xdj[i+1]; ++j)
                    cout << i+1 << ' ' << gr.Adj[j]+1 << ' ' << gr.Wgt[j] << '\n';
        }
    }

private:
    int read_mtx_to_crs(const char* filename) {
        /* variables */
        int n, i, row, col, nz_size, curr;
        int *edge_num, *last_el, *row_a, *col_a;
        T val, *val_a;
        FILE *file;
        string str;
        std::ifstream ifstream;

        /* mtx correctness check */
        if ((file = fopen(filename, "r")) == NULL) {
            printf("Cannot open file\n");
            return 1;
        }
        if (mm_read_banner(file, &(matcode))) {
            return 1;
        }
        if (mm_read_mtx_crd_size(file, &v, &n, &nz)) {
            return 1;
        }
        if (mm_is_complex(matcode) || mm_is_array(matcode)) {
            printf("This application doesn't support %s", mm_typecode_to_str(matcode));
            return 1;
        }
        if (n != v) {
            printf("Is not a square matrix\n");
            return 1;
        }
        fclose(file);

        /* Allocating memmory to store adjacency list */
        last_el  = new int[v];
        edge_num = new int[v];
        if (mm_is_symmetric(matcode)) {
            row_a = new int[2 * nz];
            col_a = new int[2 * nz];
            val_a = new   T[2 * nz];
        }
        else {
            row_a = new int[nz];
            col_a = new int[nz];
            val_a = new   T[nz];
        }
        for (i = 0; i < v; i++) {
            edge_num[i] = 0;
        }

        /* Saving value of nz so we can change it */
        nz_size = nz;

        /* Reading file to count degrees of each vertex */
        //ifstream.open(filename);
        std::ios_base::sync_with_stdio(false); // input acceleration
        ifstream.open(filename);
        do {
            std::getline(ifstream, str);
        } while (str[0] == '%');
        curr = 0;
        for(i = 0; i < nz_size; i++) {
            ifstream >> row >> col >> val;
            row--;
            col--;
            if (row == col) {
                nz--;
                continue; //we don't need loops
            }
            row_a[curr] = row;
            col_a[curr] = col;
            val_a[curr++] = val;
            ++edge_num[row];
            if (mm_is_symmetric(matcode)) {
                ++edge_num[col];
                ++nz;
                row_a[curr] = col;
                col_a[curr] = row;
                val_a[curr++] = val;
            }
        }
        std::ios_base::sync_with_stdio(true); // restoring the state
        ifstream.close();
        
        /* Creating CRS arrays */
        Adj = new int[nz];
        Xdj = new int[v+1];
        Wgt = new   T[nz];

        /* Writing data in Xadj and last_el */
        Xdj[0] = 0;
        for(i = 0; i < v; i++) {
            Xdj[i+1] = Xdj[i] + edge_num[i];
            last_el[i] = Xdj[i];
        }

        /* Reading file to write it's content in crs */
        for(i = 0; i < nz; i++) {
            Adj[last_el[row_a[i]]] = col_a[i];
            Wgt[last_el[row_a[i]]++] = val_a[i];
        }

        delete[] edge_num;
        delete[] last_el;
        delete[] row_a;
        delete[] col_a;
        delete[] val_a;
        return 0;
    }
};

#endif
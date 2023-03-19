#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <cstring>
#include "mmio.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

template <typename WgtType>
class spMatrix {
public:
    int* Col = NULL;
    int* Rst = NULL;
    WgtType* Val = (WgtType*)NULL;
    MM_typecode matcode;
    int m;
    int n;
    int nz;

    spMatrix(const char *filename, bool crs) {
        if(!crs && read_mtx_to_crs(filename)) {
            cout << "Can't read MTX from file\n";
            throw "Can't read MTX from file";
        }
        if (crs && read_crs_to_crs(filename)) {
            cout << "Can't read CRS from file\n";
            throw "Can't read CRS from file";
        }
    }

    spMatrix() {}

    spMatrix(size_t _m, size_t _n, size_t _nz): m(_m), n(_n), nz(_nz) {
        Col = new int[nz];
        Rst = new int[m+1]();
        Val = new WgtType[nz]();
    }

    spMatrix(const spMatrix &copy): m(copy.m), n(copy.n), nz(copy.nz) {
        for (int i = 0; i < 4; ++i)
            matcode[i] = copy.matcode[i];
        Col = new int[nz];
        memcpy(Col, copy.Col, nz*sizeof(int));
        Rst = new int[m+1];
        memcpy(Rst, copy.Rst, (m+1)*sizeof(int));
        if (copy.Val != NULL) {
            Val = new WgtType[nz];
            memcpy(Val, copy.Val, nz*sizeof(WgtType));
        }
    }

    spMatrix(spMatrix &&mov): m(mov.m), n(mov.n), nz(mov.nz) {
        for (int i = 0; i < 4; ++i)
            matcode[i] = mov.matcode[i];
        Col = mov.Col;
        Rst = mov.Rst;
        Val = mov.Val;

        mov.Col = NULL;
        mov.Rst = NULL;
        mov.Val = (WgtType*)NULL;
    }

    ~spMatrix() {
        if (Col)
            delete[] Col;
        if (Rst)
            delete[] Rst;
        if (Val)
            delete[] Val;
        
        Col = NULL;
        Rst = NULL;
        Val = (WgtType*)NULL;
    }

    spMatrix& operator=(const spMatrix &copy) {
        if (this == copy)
            return *this;

        for (int i = 0; i < 4; ++i)
            matcode[i] = copy.matcode[i];
        if (m != copy.m) {
            if (Rst)
                delete[] Rst;
            Rst = new int[m+1];
            memcpy(Rst, copy.Rst, (m+1)*sizeof(int));
        }
        if (nz != copy.nz) {
            if (Col)
                delete[] Col;
            if (Val)
                delete[] Val;

            Col = new int[nz];
            memcpy(Col, copy.Col, nz*sizeof(int));
            if (copy.Val != NULL) {
                Val = new WgtType[nz];
                memcpy(Val, copy.Val, nz*sizeof(WgtType));
            }
        }
        m  = copy.m;
        n  = copy.n;
        nz = copy.nz;


        return *this;
    }

    spMatrix& operator=(spMatrix &&mov) {
        if (this == &mov)
            return *this;

        m = mov.m;
        n = mov.n;
        for (int i = 0; i < 4; ++i)
            matcode[i] = mov.matcode[i];
        if (Col)
            delete[] Col;
        if (Rst)
            delete[] Rst;
        if (Val)
            delete[] Val;
        Col = mov.Col;
        Rst = mov.Rst;
        Val = mov.Val;

        mov.Col = NULL;
        mov.Rst = NULL;
        mov.Val = (WgtType*)NULL;

        m  = mov.m;
        n  = mov.n;
        nz = mov.nz;

        return *this;
    }

    void print_crs() {
        std::cout << m << ' ' << nz << '\n';
        if (Val == NULL) {
            for (int i = 0; i < m; ++i)
                for (int j = Rst[i]; j < Rst[i+1]; ++j)
                    std::cout << i+1 << ' ' << Col[j]+1 << '\n';
        }
        else {
            for (int i = 0; i < m; ++i)
                for (int j = Rst[i]; j < Rst[i+1]; ++j)
                    std::cout << i+1 << ' ' << Col[j]+1 << ' ' << Val[j] << '\n';
        }
    }

    void print_dense() {
        for (int i = 0; i < m; ++i) {
            int k = 0;
            for (int j = Rst[i]; j < Rst[i+1]; ++j, ++k) {
                while (k < Col[j]) {
                    cerr << 0 << ' ';
                    ++k;
                }
                cerr << Val[j] << ' ';
            }
            while (k < n) {
                cerr << 0 << ' ';
                ++k;
            }
            cerr << '\n';
        }
        cerr << '\n';
    }

    void print_into_file(const char *filename) {
        std::ios_base::sync_with_stdio(false); // input acceleration
        std::ofstream ofstream(filename);

        ofstream << m << ' ' << nz << ' ' << matcode << '\n';
        for (int i = 0; i < nz; ++i)
            ofstream << Col[i] << ' ';
        ofstream << '\n';
        for (int i = 0; i < nz; ++i)
            ofstream << Val[i] << ' ';
        ofstream << '\n';
        for (int i = 0; i <= m; ++i)
            ofstream << Rst[i] << ' ';
        ofstream << '\n';

        std::ios_base::sync_with_stdio(true); // restoring the state
        ofstream.close();
    }

private:
    int read_mtx_to_crs(const char* filename) {
        /* variables */
        int n, i, row, col, nz_size, curr;
        int *edge_num, *last_el, *row_a, *col_a;
        WgtType val, *val_a;
        FILE *file;
        std::string str;
        std::ifstream ifstream;

        /* mtx correctness check */
        if ((file = fopen(filename, "r")) == NULL) {
            printf("Cannot open file\n");
            return 1;
        }
        if (mm_read_banner(file, &(matcode))) {
            return 1;
        }
        if (mm_read_mtx_crd_size(file, &m, &n, &nz)) {
            return 1;
        }
        if (mm_is_complex(matcode) || mm_is_array(matcode)) {
            printf("This application doesn't support %s", mm_typecode_to_str(matcode));
            return 1;
        }
        if (n != m) {
            printf("Is not a square matrix\n");
            return 1;
        }
        fclose(file);

        /* Allocating memmory to store adjacency list */
        last_el  = new int[m];
        edge_num = new int[m];
        if (mm_is_symmetric(matcode)) {
            row_a = new int[2 * nz];
            col_a = new int[2 * nz];
            val_a = new WgtType[2 * nz];
        }
        else {
            row_a = new int[nz];
            col_a = new int[nz];
            val_a = new WgtType[nz];
        }
        for (i = 0; i < m; i++) {
            edge_num[i] = 0;
        }

        /* Saving value of nz so we can change it */
        nz_size = nz;

        /* Reading file to count degrees of each vertex */
        std::ios_base::sync_with_stdio(false);  // input acceleration
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
        Col = new int[nz];
        Rst = new int[m+1];
        Val = new   WgtType[nz];

        /* Writing data in Xadj and last_el */
        Rst[0] = 0;
        for(i = 0; i < m; i++) {
            Rst[i+1] = Rst[i] + edge_num[i];
            last_el[i] = Rst[i];
        }

        /* Reading file to write it's content in crs */
        for(i = 0; i < nz; i++) {
            Col[last_el[row_a[i]]] = col_a[i];
            Val[last_el[row_a[i]]++] = val_a[i];
        }

        delete[] edge_num;
        delete[] last_el;
        delete[] row_a;
        delete[] col_a;
        delete[] val_a;
        return 0;
    }

    int read_crs_to_crs(const char *filename) {
        std::ios_base::sync_with_stdio(false);
        std::ifstream ifstream(filename);
        if (!ifstream.is_open())
            return -1;

        ifstream >> m >> nz >> matcode;
        Col = new int[nz];
        Val = new   WgtType[nz];
        Rst = new int[m+1];
        for (int i = 0; i < nz; ++i)
            ifstream >> Col[i];
        for (int i = 0; i < nz; ++i)
            ifstream >> Val[i];
        for (int i = 0; i < m+1; ++i)
            ifstream >> Rst[i];

        ifstream.close();
        std::ios_base::sync_with_stdio(true);
        return 0;
    }
};

#endif
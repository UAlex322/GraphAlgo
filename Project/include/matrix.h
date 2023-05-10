#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
#include "mmio.h"

struct GraphInfo {
    std::string graphName;
    std::string graphPath;
    std::string logPath;
    std::string format;
};

template <typename ValT>
class spMtx {
public:
    size_t m = 0;
    size_t n = 0;
    size_t nz = 0;
    size_t capacity = 0;
    MM_typecode matcode;
    int* Rst = nullptr;
    int* Col = nullptr;
    ValT* Val = nullptr;

    spMtx(const char *filename, const std::string &format) {
        if (format == "mtx" && read_mtx_to_crs(filename)) {
            std::cout << "Can't read MTX from file\n";
            throw "Can't read MTX from file";
        } else if (format == "crs" && read_crs_to_crs(filename)) {
            std::cout << "Can't read CRS from file\n";
            throw "Can't read CRS from file";
        } else if (format == "bin" && read_bin_to_crs(filename)) {
            std::cout << "Can't read BIN from file\n";
            throw "Can't read BIN from file";
        } else if (format == "graph" && read_graph_to_crs(filename)) {
            std::cout << "Can't read GRAPH from file\n";
            throw "Can't read GRAPH from file";
        } else if (format == "rmat" && read_rmat_to_crs(filename)) {
            std::cout << "Can't read RMAT from file\n";
            throw "Can't read RMAT from file";
        }

        capacity = nz;
    }

    spMtx() {}

    spMtx(size_t _m, size_t _n): m(_m), n(_n) {
        Rst = new int[m+1]();
    }

    spMtx(size_t _m, size_t _n, size_t _nz): m(_m), n(_n), nz(_nz) {
        Rst = new int[m+1]();
        Col = new int[nz];
        Val = new ValT[nz]();
        capacity = nz;
    }

    spMtx(const spMtx &copy): m(copy.m), n(copy.n), nz(copy.nz), capacity(copy.capacity) {
        for (int i = 0; i < 4; ++i)
            matcode[i] = copy.matcode[i];
        Col = new int[nz];
        memcpy(Col, copy.Col, nz*sizeof(int));
        Rst = new int[m+1];
        memcpy(Rst, copy.Rst, (m+1)*sizeof(int));
        if (copy.Val != nullptr) {
            Val = new ValT[nz];
            memcpy(Val, copy.Val, nz*sizeof(ValT));
        }
    }

    spMtx(spMtx &&mov): m(mov.m), n(mov.n), nz(mov.nz), capacity(mov.capacity) {
        for (int i = 0; i < 4; ++i)
            matcode[i] = mov.matcode[i];
        Col = mov.Col;
        Rst = mov.Rst;
        Val = mov.Val;

        mov.Col = nullptr;
        mov.Rst = nullptr;
        mov.Val = nullptr;
    }

    ~spMtx() {
        if (Col)
            delete[] Col;
        if (Rst)
            delete[] Rst;
        if (Val)
            delete[] Val;
        
        Col = nullptr;
        Rst = nullptr;
        Val = nullptr;
    }

    // TODO: fix memory leaks
    void resizeVals(size_t newNz) {
        if (newNz > capacity) {
            if (Col != nullptr)
                delete Col;
            if (Val != nullptr)
                delete Val;
            Col = new  int[newNz];
            Val = new ValT[newNz];
            capacity = newNz;
        }
        nz = newNz;
    }

    void resizeRows(size_t newM) {
        if (m != newM) {
            if (Rst != nullptr)
                delete Rst;
            Rst = new int[newM + 1]();
            m = newM;
        }
    }

    // Копирование структуры матрицы без копирования значений
    template <typename ValT2>
    void copyPattern(const spMtx<ValT2> &source) {
        resizeRows(source.m);
        memcpy(Rst, source.Rst, (m + 1) * sizeof(int));
        n = source.n;

        resizeVals(source.nz);
        memcpy(Col, source.Col, nz * sizeof(int));
    }

    spMtx& operator=(const spMtx &copy) {
        if (this == &copy)
            return *this;

        for (int i = 0; i < 4; ++i)
            matcode[i] = copy.matcode[i];
        if (m != copy.m) {
            if (Rst)
                delete[] Rst;
            Rst = new int[copy.m + 1];
        }
        memcpy(Rst, copy.Rst, (copy.m + 1) * sizeof(int));
        if (capacity < copy.nz) {
            if (Col)
                delete[] Col;
            if (Val)
                delete[] Val;

            Col = new int[copy.nz];
            Val = new ValT[copy.nz];
            capacity = copy.nz;
        }
        memcpy(Col, copy.Col, copy.nz * sizeof(int));
        memcpy(Val, copy.Val, copy.nz * sizeof(ValT));

        m = copy.m;
        n  = copy.n;
        nz = copy.nz;

        return *this;
    }

    spMtx& operator=(spMtx &&mov) {
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

        mov.Col = nullptr;
        mov.Rst = nullptr;
        mov.Val = nullptr;

        m  = mov.m;
        n  = mov.n;
        nz = mov.nz;
        capacity = mov.capacity;

        return *this;
    }

    spMtx extractRows(size_t begin, size_t end) const {
        spMtx result(end - begin, n, Rst[end] - Rst[begin]);

        for (size_t i = 0; i <= end - begin; ++i)
            result.Rst[i] = Rst[i + begin] - Rst[begin];
        memcpy(result.Col, Col + Rst[begin], (Rst[end] - Rst[begin]) * sizeof(int));
        memcpy(result.Val, Val + Rst[begin], (Rst[end] - Rst[begin]) * sizeof(ValT));
        memcpy(result.matcode, matcode, sizeof(MM_typecode));

        return result;
    }

    bool operator==(const spMtx &other) const {
        if (m != other.m || n != other.n || nz != other.nz)
            return false;
        for (size_t i = 0; i <= m; ++i)
            if (Rst[i] != other.Rst[i])
                return false;
        for (size_t j = 0; j < nz; ++j)
            if (Col[j] != other.Col[j] || Val[j] != other.Val[j])
                return false;
        return true;
    }

    void print_crs() const {
        std::cout << m << ' ' << nz << '\n';
        if (Val == nullptr) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst[i]; j < Rst[i+1]; ++j)
                    std::cout << i+1 << ' ' << Col[j]+1 << '\n';
        }
        else {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst[i]; j < Rst[i+1]; ++j)
                    std::cout << i+1 << ' ' << Col[j]+1 << ' ' << Val[j] << '\n';
        }
    }

    void print_dense() const {
        for (size_t i = 0; i < m; ++i) {
            size_t k = 0;
            for (size_t j = Rst[i]; j < Rst[i+1]; ++j, ++k) {
                while (k < Col[j]) {
                    std::cerr << 0 << ' ';
                    ++k;
                }
                std::cerr << Val[j] << ' ';
            }
            while (k < n) {
                std::cerr << 0 << ' ';
                ++k;
            }
            std::cerr << '\n';
        }
        std::cerr << '\n';
    }

    int write_crs_to_bin(const char *filename) {
        FILE *fp = fopen(filename, "wb");
        if (fp == NULL)
            return -1;
        
        fwrite(matcode, 1, 1, fp);
        fwrite(matcode + 1, 1, 1, fp);
        fwrite(matcode + 2, 1, 1, fp);
        fwrite(matcode + 3, 1, 1, fp);
        fwrite(&m, sizeof(size_t), 1, fp);
        fwrite(&n, sizeof(size_t), 1, fp);
        fwrite(&nz, sizeof(size_t), 1, fp);
        fwrite(Rst, sizeof(int), m+1, fp);
        fwrite(Col, sizeof(int), nz, fp);
        fwrite(Val, sizeof(ValT), nz, fp);

        fclose(fp);
        return 0;
    }

private:
    int read_mtx_to_crs(const char* filename) {
        /* variables */
        size_t row, col, nz_size, curr;
        int *edge_num, *last_el, *row_a, *col_a, m_int, n_int, nz_int;
        ValT val, *val_a;
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
        if (mm_read_mtx_crd_size(file, &m_int, &n_int, &nz_int)) {
            return 1;
        }
        m = (size_t)m_int;
        n = (size_t)n_int;
        nz = (size_t)nz_int;
        if (mm_is_complex(matcode) || mm_is_array(matcode)) {
            printf("This application doesn't support %s", mm_typecode_to_str(matcode));
            return 1;
        }
        if (m != n) {
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
            val_a = new ValT[2 * nz];
        }
        else {
            row_a = new int[nz];
            col_a = new int[nz];
            val_a = new ValT[nz];
        }
        for (size_t i = 0; i < m; i++) {
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
        if (mm_is_pattern(matcode)) {
            for(size_t i = 0; i < nz_size; i++) {
                ifstream >> row >> col;
                row--;
                col--;
                if (row == col) {
                    nz--;
                    continue; //we don't need loops
                }
                row_a[curr] = row;
                col_a[curr++] = col;
                ++edge_num[row];
                if (mm_is_symmetric(matcode)) {
                    ++edge_num[col];
                    ++nz;
                    row_a[curr] = col;
                    col_a[curr++] = row;
                }
            }
        }
        else {
            for (size_t i = 0; i < nz_size; i++) {
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
        }
        std::ios_base::sync_with_stdio(true); // restoring the state
        ifstream.close();
        
        /* Creating CRS arrays */
        Col = new int[nz];
        Rst = new int[m+1];
        Val = new ValT[nz];

        /* Writing data in Rst and last_el */
        Rst[0] = 0;
        for(size_t i = 0; i < m; i++) {
            Rst[i+1] = Rst[i] + edge_num[i];
            last_el[i] = Rst[i];
        }

        /* Reading file to write it's content in crs */
        if (mm_is_pattern(matcode)) {
            for (size_t i = 0; i < nz; ++i) {
                Col[last_el[row_a[i]]] = col_a[i];
                Val[i] = 1;
            }
        } else {
            for (size_t i = 0; i < nz; ++i) {
                Col[last_el[row_a[i]]] = col_a[i];
                Val[last_el[row_a[i]]++] = val_a[i];
            }
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
        Val = new ValT[nz];
        Rst = new int[m+1];
        for (size_t i = 0; i < nz; ++i)
            ifstream >> Col[i];
        for (size_t i = 0; i < nz; ++i)
            ifstream >> Val[i];
        for (size_t i = 0; i < m+1; ++i)
            ifstream >> Rst[i];

        ifstream.close();
        std::ios_base::sync_with_stdio(true);
        return 0;
    }

    int read_bin_to_crs(const char *filename) {
        FILE *fp = fopen(filename, "rb");
        if (fp == NULL)
            return -1;

        fread(matcode, 1, 1, fp);
        fread(matcode + 1, 1, 1, fp);
        fread(matcode + 2, 1, 1, fp);
        fread(matcode + 3, 1, 1, fp);
        fread(&m, sizeof(size_t), 1, fp);
        fread(&n, sizeof(size_t), 1, fp);
        fread(&nz, sizeof(size_t), 1, fp);


        Rst = new int[m+1];
        Col = new int[nz];
        Val = new ValT[nz];
        fread(Rst, sizeof(int), m+1, fp);
        fread(Col, sizeof(int), nz, fp);
        fread(Val, sizeof(ValT), nz, fp);

        fclose(fp);
        return 0;
    }

    int read_graph_to_crs(const char *filename) {
        std::ifstream ifstr(filename, std::ios::in);
        if (!ifstr.is_open())
            return -1;
        std::ios::sync_with_stdio(false);

        ifstr >> m >> nz;
        n = m;
        Rst = new int[m+1];
        Col = new int[nz];
        Val = new ValT[nz];
        Rst[0] = 0;

        std::string s;
        size_t j = 0;
        for (size_t i = 0; i < m; ++i) {
            std::getline(ifstr, s);
            std::istringstream iss(s);
            while (iss >> Col[j]) {
                Val[j] = 1;
                j++;
            }
            Rst[i+1] = j;
        }
        matcode[2] = 'I';

        std::ios::sync_with_stdio(true);
        ifstr.close();
        return 0;
    }

    int read_rmat_to_crs(const char *filename) {
        std::ifstream ifstr(filename, std::ios::in);
        if (!ifstr.is_open())
            return -1;
        std::ios::sync_with_stdio(false);

        ifstr >> m >> n >> nz;
        Rst = new int[m+1]();
        int *TmpX = new int[nz];
        int *TmpY = new int[nz];
        Col = new int[2*nz];
        Val = new ValT[2*nz];

        size_t x, y;
        matcode[2] = 'I';
        for (size_t j = 0; j < nz; ++j) {
            ifstr >> x >> y;
            TmpX[j] = x;
            TmpY[j] = y;
            ++Rst[x+1];
            ++Rst[y+1];
        }
        Rst[0] = 0;
        for (size_t i = 1; i < m; ++i)
            Rst[i+1] += Rst[i];
        for (size_t j = 0; j < nz; ++j) {
            x = TmpX[j];
            y = TmpY[j];
            Col[Rst[x]++] = y;
            Col[Rst[y]++] = x;
        }
        for (size_t i = m; i > 0; --i)
            Rst[i] = Rst[i-1];
        Rst[0] = 0;
        for (size_t j = 0; j < 2*nz; ++j)
            Val[j] = 1;
        nz *= 2;

        delete[] TmpX;
        delete[] TmpY;
        std::ios::sync_with_stdio(true);
        ifstr.close();
        return 0;
    }
};

template <typename ValT>
class denseMtx {
public:
    size_t m = 0;
    size_t n = 0;
    size_t capacity = 0;
    ValT *Val = nullptr;

    denseMtx() {}

    denseMtx(size_t _m, size_t _n) : m(_m), n(_n) {
        capacity = m*n;
        Val = new ValT[capacity];
    }
    denseMtx(const denseMtx &copy) : m(copy.m), n(copy.n) {
        capacity = m*n;
        Val = new ValT[capacity];
        memcpy(Val, copy.Val, capacity * sizeof(ValT));
    }
    denseMtx(denseMtx &&mov) : m(mov.m), n(mov.n) {
        capacity = m*n;
        Val = mov.Val;
        mov.Val = nullptr;
    }
    template <typename ValT2>
    denseMtx(const spMtx<ValT2> &copy) : m(copy.m), n(copy.n) {
        capacity = m*n;
        Val = new ValT[capacity]();
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < copy.m; ++i) {
            for (size_t k = copy.Rst[i]; k < copy.Rst[i+1]; ++k) {
                size_t j = copy.Col[k];
                Val[n * i + j] = (ValT)(copy.Val[k]);
            }
        }
    }
    denseMtx& operator=(const denseMtx &copy) {
        if (this != &copy)
            return *this;

        m = copy.m;
        n = copy.n;
        if (capacity < m*n) {
            if (Val != nullptr)
                delete[] Val;
            capacity = m*n;
            Val = new ValT[capacity];
        }
        memcpy(Val, copy.Val, m*n*sizeof(ValT));

        return *this;
    }
    template <typename ValT2>
    denseMtx& operator=(const spMtx<ValT2> &copy) {
        m = copy.m;
        n = copy.n;
        if (capacity < m*n) {
            if (Val != nullptr)
                delete[] Val;
            capacity = m*n;
            Val = new ValT[capacity];
        }
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < copy.m; ++i) {
            for (int k = copy.Rst[i]; k < copy.Rst[i+1]; ++k) {
                size_t j = (size_t)copy.Col[k];
                Val[n * i + j] = (ValT)(copy.Val[k]);
            }
        }
        return *this;
    }
    denseMtx& operator=(denseMtx &&mov) {
        m = mov.m;
        n = mov.n;
        capacity = mov.capacity;
        if (Val != nullptr)
            delete[] Val;
        Val = mov.Val;
        mov.Val = nullptr;

        return *this;
    }
    ~denseMtx() {
        delete[] Val;
        Val = nullptr;
    }

    void resize(size_t newM, size_t newN) {
        m = newM;
        n = newN;
        if (capacity < m*n) {
            if (Val != nullptr)
                delete[] Val;
            Val = new ValT[m*n];
            capacity = m*n;
        }
    }
    void print() {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j)
                std::cout << std::setw(5) << Val[n * i + j] << ' ';
            std::cout << '\n';
        }
        std::cout << '\n';
    }
};

// C += A .* B
template <typename T>
void fuseEWiseMultAdd(const denseMtx<T> &A, const denseMtx<T> &B, const denseMtx<T> &C) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m * A.n; ++i)
        C.Val[i] += A.Val[i] * B.Val[i];
}

// C<M> = A .* B
template <typename T, typename U>
void eWiseMult(const denseMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, const denseMtx<T> &C) {
    memset(C.Val, 0, C.m * C.n * sizeof(T));
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < M.m; ++i) {
        for (size_t j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
            size_t idx = C.n * i + M.Col[j];
            C.Val[idx] = A.Val[idx] * B.Val[idx];
        }
    }
}

template <typename T>
denseMtx<T> transpose(const denseMtx<T> &A) {
    denseMtx<T> AT(A.n, A.m);
    size_t block_size = 64;

    for (size_t i = 0; i < A.m; i += block_size) {
        for (size_t j = 0; j < A.n; j += block_size) {
            size_t pmax = std::min(A.m, i + block_size);
            size_t qmax = std::min(A.n, j + block_size);
            for (size_t p = i; p < pmax; ++p)
                for (size_t q = j; q < qmax; ++q)
                    AT.Val[AT.n * q + p] = A.Val[A.n * p + q];
        }
    }

    return AT;
}

// C<M> = A * B
template <typename T, typename U>
void mxmm(const spMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, denseMtx<T> &C) {
    denseMtx<T> Ccopy(C.m, C.n);
    memset(Ccopy.Val, 0, C.m * C.n * sizeof(T));
    denseMtx<T> BT = transpose(B);
    T zero = (T)0;

// #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < M.m; ++i) {
        for (size_t q = M.Rst[i]; q < M.Rst[i+1]; ++q) {
            size_t j = M.Col[q];
            T *b_row = BT.Val + BT.n * j;
            T dotpr = zero;

            for (size_t k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
                dotpr += A.Val[k] * b_row[A.Col[k]];
            }
            Ccopy.Val[C.n * i + j] = dotpr;
        }
    }
    C = std::move(Ccopy);
}



template <typename ValT2, typename ValT1>
spMtx<ValT2> convertType(const spMtx<ValT1> &source) {
    spMtx<ValT2> result;

    result.copyPattern(source);
    for (size_t j = 0; j < source.nz; ++j)
        result.Val[j] = (ValT2)(source.Val[j]);

    return result;
}


spMtx<int> generate_mask(const size_t n, const size_t min_deg, const size_t max_deg) {
    std::set<int> positions;
    std::uniform_int_distribution<int> deg_distr(min_deg, max_deg);
    std::uniform_int_distribution<int> col_distr(0, n - 1);
    std::mt19937 generator{std::random_device{}()};
    spMtx<int> mask;
    mask.Rst = new int[n+1];
    mask.m = mask.n = n;

    mask.Rst[0] = 0;
    for (size_t i = 0; i < n; ++i)
        mask.Rst[i+1] = mask.Rst[i] + deg_distr(generator);
    mask.nz = mask.Rst[n];
    mask.Col = new int[mask.nz];

    size_t j = 0;
    for (size_t i = 0; i < mask.n; ++i) {
        size_t deg = mask.Rst[i+1] - mask.Rst[i];
        while (positions.size() < deg)
            positions.insert(col_distr(generator));
        for (const int column : positions)
            mask.Col[j++] = column;
        positions.clear();
    }

    return mask;
}

template<typename T>
void full_mask(spMtx<T> &mask, const size_t n) {
    mask.m = n;
    mask.nz = n*n;
    mask.Rst = new int[n+1];
    mask.Col = new int[n*n];

    for (size_t i = 0; i <= n; ++i)
        mask.Rst[i] = i*n;
    size_t curr_pos = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            mask.Col[curr_pos++] = j;
    }
}

template <typename T>
spMtx<int> build_adjacency_matrix(const spMtx<T> &Gr) {
    spMtx<int> Res;
    Res.m = Gr.m;
    Res.n = Gr.n;
    Res.nz = Gr.nz;
    Res.Rst = new int[Gr.m + 1];
    Res.Col = new int[Gr.nz];
    Res.Val = new int[Gr.nz];

    std::memcpy(Res.Col, Gr.Col, Gr.nz*sizeof(int));
    std::memcpy(Res.Rst, Gr.Rst, (Gr.m + 1)*sizeof(int));
    std::memcpy(Res.matcode, Gr.matcode, 4);
    Res.matcode[2] = 'I';

    for (size_t i = 0; i < Gr.nz; ++i)
        Res.Val[i] = 1;

    return Res;
}

template <typename T>
spMtx<T> build_symm_from_lower(const spMtx<T> &Low) {
    spMtx<T> Res(Low.m, Low.n, 2*Low.nz);
    spMtx<T> Upp = transpose(Low);
    size_t jl = 0;
    size_t ju = 0;
    size_t jr = 0;

    for (size_t i = 0; i < Low.m; ++i) {
        size_t xl = Low.Rst[i+1];
        size_t xu = Upp.Rst[i+1];

        while (jl < xl && ju < xu) {
            if (Low.Col[jl] < Upp.Col[ju]) {
                Res.Col[jr] = Low.Col[jl];
                Res.Val[jr++] = Low.Val[jl++];
            }
            else {
                Res.Col[jr] = Upp.Col[ju];
                Res.Val[jr++] = Upp.Val[ju++];
            }
        }
        while (jl < xl) {
            Res.Col[jr] = Low.Col[jl];
            Res.Val[jr++] = Low.Val[jl++];
        }
        while (ju < xu) {
            Res.Col[jr] = Upp.Col[ju];
            Res.Val[jr++] = Upp.Val[ju++];
        }
        Res.Rst[i+1] = jr;
    }

    return Res;
}

template <typename T>
spMtx<T> extract_lower_triangle(const spMtx<T> &Gr) {
    spMtx<T> Res;

    Res.m = Gr.m;
    Res.n = Gr.n;
    Res.Rst = new int[Gr.m + 1];
    Res.Rst[0] = 0;

    for (size_t i = 0; i < Gr.m; ++i) {
        int r = Gr.Rst[i];
        while (r < Gr.Rst[i+1] && Gr.Col[r] < i)
            ++r;
        Res.Rst[i+1] = Res.Rst[i] + (r - Gr.Rst[i]);
    }

    Res.nz = Res.Rst[Res.m];
    Res.Col = new int[Res.nz];
    Res.Val = new   T[Res.nz];

    for (size_t i = 0; i < Gr.m; ++i) {
        size_t row_len = Res.Rst[i+1] - Res.Rst[i];
        std::memcpy(Res.Col + Res.Rst[i], Gr.Col + Gr.Rst[i], row_len*sizeof(int));
        std::memcpy(Res.Val + Res.Rst[i], Gr.Val + Gr.Rst[i], row_len*sizeof(T));
    }
    
    return Res;
}

template <typename T>
spMtx<T> extract_upper_triangle(const spMtx<T> &Gr) {
    spMtx<T> Res;

    Res.m = Gr.m;
    Res.Rst = new int[Gr.m + 1];
    Res.Rst[0] = 0;

    for (size_t i = 0; i < Gr.m; ++i) {
        size_t r = Gr.Rst[i];
        while (r < Gr.Rst[i+1] && Gr.Col[r] <= i)
            ++r;
        Res.Rst[i+1] = Res.Rst[i] + (Gr.Rst[i+1] - r);
    }

    Res.nz = Res.Rst[Res.m];
    Res.Col = new int[Res.nz];
    Res.Val = new   T[Res.nz];

    for (size_t i = 0; i < Gr.m; ++i) {
        int row_len = Res.Rst[i+1] - Res.Rst[i];
        int row_offset = Gr.Rst[i+1] - row_len;
        std::memcpy(Res.Col + Res.Rst[i], Gr.Col + row_offset, row_len*sizeof(int));
        std::memcpy(Res.Val + Res.Rst[i], Gr.Val + row_offset, row_len*sizeof(T));
    }

    return Res;
}

spMtx<int> generate_adjacency_matrix(const size_t n, const size_t min_deg, const size_t max_deg) {
    spMtx<int> Res = generate_mask(n, min_deg, max_deg);

    Res.Val = new int[Res.nz];
    for (size_t j = 0; j < Res.nz; ++j)
        Res.Val[j] = 1;

    return build_symm_from_lower(extract_lower_triangle(Res));
}

// template <typename T, typename U>
// spMtx<T> eWiseAdd(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M) {
//     if (A.m != B.m || A.n != B.n)
//         throw -1;
// 
//     spMtx<T> C(A.m, A.n);
//     const T zero = (T)0;
// 
// #pragma omp parallel 
//     {
//         T *row = new T[A.n];
// 
// #pragma omp for schedule(dynamic)
//         for (int i = 0; i < A.m; ++i) {
//             int m_pos = M.Rst[i], m_max = M.Rst[i+1];
//             int col_cnt = 0;
// 
//             for (int j = m_pos; j < m_max; ++j)
//                 row[M.Col[j]] = zero;
//             for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j)
//                 row[A.Col[j]] += A.Val[j];
//             for (int j = B.Rst[i]; j < B.Rst[i+1]; ++j)
//                 row[A.Col[j]] += B.Val[j];
//             for (int j = m_pos; j < m_max; ++j)
//                 if (row[M.Col[j]])
//                     ++col_cnt;
// 
//             C.Rst[i+1] = col_cnt;
//         }
// 
//         delete[] row;
//     }
// 
//     for (int i = 0; i < A.m; ++i)
//         C.Rst[i+1] += C.Rst[i];
//     C.resizeVals(C.Rst[A.m]);
//     
// #pragma omp parallel 
//     {
//         T *row = new T[A.n];
// 
// #pragma omp for schedule(dynamic)
//         for (int i = 0; i < A.m; ++i) {
//             int m_pos = M.Rst[i], m_max = M.Rst[i+1];
//             int c_pos = C.Rst[i];
// 
//             for (int j = m_pos; j < m_max; ++j)
//                 row[M.Col[j]] = zero;
//             for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j)
//                 row[A.Col[j]] += A.Val[j];
//             for (int j = B.Rst[i]; j < B.Rst[i+1]; ++j)
//                 row[A.Col[j]] += B.Val[j];
//             for (int j = m_pos; j < m_max; ++j)
//                 if (row[M.Col[j]]) {
//                     C.Col[c_pos] = M.Col[j];
//                     C.Val[c_pos++] = row[M.Col[j]];
//                 }
//         }
// 
//         delete[] row;
//     }
// 
//     return C;
// }

template <typename T>
spMtx<T> eWiseAdd(const spMtx<T> &A, const spMtx<T> &B) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);
    // индекс, начиная с которого в строке идут элементы только из одной матрицы
    int *rowMergeEnd = new int[A.m]();

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int colCnt = 0;
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        while (aIdx < A.Rst[i+1] && bIdx < B.Rst[i+1]) {
            if (A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            else if (A.Col[aIdx] > B.Col[bIdx])
                ++bIdx;
            else
                ++aIdx, ++bIdx;
            ++colCnt;
        }
        rowMergeEnd[i] = colCnt;
        colCnt += (A.Rst[i+1] - aIdx) + (B.Rst[i+1] - bIdx);
        C.Rst[i+1] = colCnt;
    }

    C.Rst[0] = 0;
    for (size_t i = 0; i < A.m; ++i) {
        C.Rst[i+1] += C.Rst[i];
        rowMergeEnd[i] += C.Rst[i];
    }
    C.resizeVals(C.Rst[A.m]);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        for (int cIdx = C.Rst[i]; cIdx < rowMergeEnd[i]; ++cIdx) {
            if (A.Col[aIdx] < B.Col[bIdx]) {
                C.Col[cIdx] = A.Col[aIdx];
                C.Val[cIdx] = A.Val[aIdx++];
            } else if (A.Col[aIdx] > B.Col[bIdx]) {
                C.Col[cIdx] = B.Col[bIdx];
                C.Val[cIdx] = B.Val[bIdx++];
            } else {
                C.Col[cIdx] = A.Col[aIdx];
                C.Val[cIdx] = A.Val[aIdx++] + B.Val[bIdx++];
            }
        }
        if (aIdx < A.Rst[i+1]) {
            memcpy(C.Col + rowMergeEnd[i], A.Col + aIdx, (C.Rst[i+1] - rowMergeEnd[i]) * sizeof(int));
            memcpy(C.Val + rowMergeEnd[i], A.Val + aIdx, (C.Rst[i+1] - rowMergeEnd[i]) * sizeof(T));
        } else {
            memcpy(C.Col + rowMergeEnd[i], B.Col + bIdx, (C.Rst[i+1] - rowMergeEnd[i]) * sizeof(int));
            memcpy(C.Val + rowMergeEnd[i], B.Val + bIdx, (C.Rst[i+1] - rowMergeEnd[i]) * sizeof(T));
        }
    }

    delete[] rowMergeEnd;
    return C;
}


template <typename T>
spMtx<T> eWiseMult(const spMtx<T> &A, const spMtx<T> &B) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1];
        int colCnt = 0;

        while (aIdx < aMax && bIdx < bMax) {
            if (A.Col[aIdx] == B.Col[bIdx])
                ++aIdx, ++bIdx, ++colCnt;
            while (aIdx < aMax && A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < A.Col[aIdx])
                ++bIdx;
        }
        C.Rst[i+1] = colCnt;
    }

    for (size_t i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.resizeVals(C.Rst[C.m]);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1];

        for (int cIdx = C.Rst[i]; cIdx < C.Rst[i+1]; ++i) {
            if (A.Col[aIdx] == B.Col[bIdx]) {
                C.Col[cIdx] = A.Col[aIdx];
                C.Val[cIdx] = A.Val[aIdx++] * B.Val[bIdx++];
            }
            while (aIdx < aMax && A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < A.Col[aIdx])
                ++bIdx;
        }
    }
    return C;
}


template <typename T, typename U>
spMtx<T> eWiseMult(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1], mMax = M.Rst[i+1];
        int colCnt = 0;

        for (int mIdx = M.Rst[i]; mIdx < mMax; ++mIdx) {
            if (A.Col[aIdx] == B.Col[bIdx] && B.Col[bIdx] == M.Col[mIdx]) {
                ++aIdx;
                ++bIdx;
                ++mIdx;
                ++colCnt;
            }
            while (aIdx < aMax && A.Col[aIdx] < M.Col[mIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < M.Col[mIdx])
                ++bIdx;
        }
        C.Rst[i+1] = colCnt;
    }

    for (size_t i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.resizeVals(C.Rst[C.m]);
    
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i], mIdx = M.Rst[i];
        for (int j = C.Rst[i]; j < C.Rst[i+1]; ++j) {
            if (A.Col[aIdx] == B.Col[bIdx] && B.Col[bIdx] == M.Col[mIdx]) {
                C.Col[j] = A.Col[aIdx];
                C.Val[j] = A.Val[aIdx++] * B.Val[bIdx++];
            } else if (A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            else
                ++bIdx;
        }
    }

    return C;
}

template <typename MatrixValT, typename ScalarT>
spMtx<MatrixValT> multScalar(const spMtx<MatrixValT> &A, const ScalarT &alpha) {
#pragma omp parallel for
    for (size_t i = 0; i < A.nz; ++i)
        A.Val[i] *= alpha;
    return A;
}


/*MSpGEMM с использованием сжатого (разрежённого) аккумулятора*/
template <typename T>
struct MCA {
    static enum {ALLOWED = 0, SET} mca_states;
    // char   *states;
    T      *values;
    size_t  len;

    MCA(size_t n) {
        values = new T[n];
        // states = new char[n];
           len = n;

        std::memset(values, 0, len * sizeof(T));
        // std::memset(states, 0, len * sizeof(char));
    }

    ~MCA() {
        delete[] values;
        // delete[] states;
    }

    inline void clear() {
        std::memset(values, 0, len * sizeof(T));
        // std::memset(states, 0, len * sizeof(char));
    }
};

template<typename T, typename U>
spMtx<T> mxmm_mca(bool isParallel, const spMtx<T> &A, const spMtx<U> &B, const spMtx<T> &M) {
    // инициализация C
    spMtx<T> C(A.m, B.n, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_mca_parallel(A, B, M, C);
    else
        _mxmm_mca_sequential(A, B, M, C);

    return C;
}

template<typename T, typename U>
void mxmm_mca(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // инициализация C
    C.resizeRows(M.m);
    C.resizeVals(M.nz);
    C.n = M.n;
    memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_mca_parallel(A, B, M, C);
    else
        _mxmm_mca_sequential(A, B, M, C);
}

template<typename T, typename U>
void _mxmm_mca_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    int mca_len = 0;
    for (size_t i = 0; i < A.m; ++i)
        if (M.Rst[i+1] - M.Rst[i] > mca_len)
            mca_len = M.Rst[i+1] - M.Rst[i];

#pragma omp parallel
    {
        MCA<T> accum(mca_len);

#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            int m_row_len = M.Rst[i+1] - M.Rst[i];
            int m_pos;

            // подсчёт i-й строки матрицы C
            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   a_val = A.Val[t];
                // прохождение строки и обработка только входящих в маску элементов
                m_pos = M.Rst[i];
                for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                    // ищем следующий входящий в маску элемент
                    while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                        ++b_pos;
                    // при нахождении накапливаем значение
                    if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                        accum.values[j] += a_val * B.Val[b_pos];
                }
            }

            // заполнение i-й строки матрицы C
            memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
            // очистка аккумулятора для следующей итерации
            memset(accum.values, 0, mca_len * sizeof(T));
        }
    }
}

template<typename T, typename U>
void _mxmm_mca_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    int mca_len = 0;
    for (size_t i = 0; i < A.m; ++i)
        if (M.Rst[i+1] - M.Rst[i] > mca_len)
            mca_len = M.Rst[i+1] - M.Rst[i];

    MCA<T> accum(mca_len);

    for (size_t i = 0; i < A.m; ++i) {
        int m_row_len = M.Rst[i+1] - M.Rst[i];
        int m_pos;

        // подсчёт i-й строки матрицы C
        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];
            // прохождение строки и обработка только входящих в маску элементов
            m_pos = M.Rst[i];
            for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                // ищем следующий входящий в маску элемент
                while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                    ++b_pos;
                // при нахождении накапливаем значение
                if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                    accum.values[j] += a_val * B.Val[b_pos];
            }
        }
        // заполнение i-й строки матрицы C
        memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
        // очистка аккумулятора для следующей итерации
        memset(accum.values, 0, mca_len * sizeof(T));
    }
}


template <typename T>
struct MSA {
    static enum {UNALLOWED = 0, ALLOWED, SET} msa_states;
    char   *state;
    T      *value;
    size_t  len;

    MSA(size_t n) {
        value = new T[n]();
        state = new char[n]();
        len = n;
    }

    ~MSA() {
        delete[] value;
        delete[] state;
    }
};

template<typename T, typename U>
void mxmm_msa(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // инициализация C
    C.resizeRows(M.m);
    C.resizeVals(M.nz);
    C.n = M.n;
    memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_msa_parallel(A, B, M, C);
    else
        _mxmm_msa_sequential(A, B, M, C);
}

template<typename T, typename U>
void mxmm_msa_cmask(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // инициализация C
    C.resizeRows(M.m);
    C.n = M.n;

    if (isParallel)
        _mxmm_msa_cmask_parallel(A, B, M, C);
    else
        _mxmm_msa_cmask_sequential(A, B, M, C);
}

template<typename T, typename U>
void _mxmm_msa_cmask_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
#pragma omp parallel
    {
        MSA<T> accum(B.n);
        
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i+1];

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];

                for (int j = b_pos; j < b_max; ++j)
                    accum.state[B.Col[j]] = MSA<T>::ALLOWED;
            }
            for (int j = m_min; j < m_max; ++j) {
                accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
            }
            C.Rst[i+1] = 0;
            for (int j = 0; j < accum.len; ++j) {
                C.Rst[i+1] += accum.state[j];
                accum.state[j] = MSA<T>::UNALLOWED;
            }
        }
    }
    C.Rst[0] = 0;
    for (int i = 1; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    if (C.Rst[A.m] > C.nz)
        C.resizeVals(C.Rst[A.m]);
    C.nz = C.Rst[A.m];

#pragma omp parallel
    {
        MSA<T> accum(B.n);
        T zero = (T)0;

#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i+1];

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   a_val = A.Val[t];

                for (int j = b_pos; j < b_max; ++j) {
                    accum.state[B.Col[j]] = MSA<T>::SET;
                    accum.value[B.Col[j]] += a_val * B.Val[j];
                }
            }

            for (int j = m_min; j < m_max; ++j) {
                accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
            }
            int c_pos = C.Rst[i];
            for (int j = 0; j < accum.len; ++j) {
                if (accum.state[j]) {
                    C.Col[c_pos] = j;
                    C.Val[c_pos++] = accum.value[j];
                }
                accum.state[j] = MSA<T>::UNALLOWED;
                accum.value[j] = zero;
            }
        }
    }
}

template<typename T, typename U>
void _mxmm_msa_cmask_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    MSA<T> accum(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];

            for (int j = b_pos; j < b_max; ++j)
                accum.state[B.Col[j]] = MSA<T>::ALLOWED;
        }
        for (int j = m_min; j < m_max; ++j) {
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
        }
        C.Rst[i+1] = 0;
        for (int j = 0; j < accum.len; ++j) {
            C.Rst[i+1] += accum.state[j];
            accum.state[j] = MSA<T>::UNALLOWED;
        }
    }
    C.Rst[0] = 0;
    for (int i = 1; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    if (C.Rst[A.m] > C.nz)
        C.resizeVals(C.Rst[A.m]);
    C.nz = C.Rst[A.m];

    T zero = (T)0;
    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                accum.state[B.Col[j]] = MSA<T>::SET;
                accum.value[B.Col[j]] += a_val * B.Val[j];
            }
        }

        for (int j = m_min; j < m_max; ++j) {
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
        }
        int c_pos = C.Rst[i];
        for (int j = 0; j < accum.len; ++j) {
            if (accum.state[j]) {
                C.Col[c_pos] = j;
                C.Val[c_pos++] = accum.value[j];
            }
            accum.state[j] = MSA<T>::UNALLOWED;
            accum.value[j] = zero;
        }
    }
}

template<typename T, typename U>
void _mxmm_msa_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    MSA<T> accum(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        // Выделение допустимых элементов аккумулятора
        for (int j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::ALLOWED;

        // Подсчёт i-й строки матрицы C
        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                int b_col = B.Col[j];
                if (accum.state[b_col] == MSA<T>::ALLOWED) {
                    accum.state[b_col] = MSA<T>::SET;
                    accum.value[b_col] = a_val * B.Val[j];
                } else if (accum.state[b_col] == MSA<T>::SET)
                    accum.value[b_col] += a_val * B.Val[j];
            }
        }

        // Заполнение строки матрицы C и очистка аккумулятора
        for (int j = m_min; j < m_max; ++j) {
            C.Val[j] = accum.value[M.Col[j]];
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
        }
    }
}


template<typename T, typename U>
void _mxmm_msa_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
#pragma omp parallel
    {
        MSA<T> accum(B.n);
        T zero = (T)0;

#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            int m_min = M.Rst[i];
            int m_max = M.Rst[i+1];

            // Выделение допустимых элементов аккумулятора
            for (int j = m_min; j < m_max; ++j)
                accum.value[M.Col[j]] = zero;

            // Подсчёт i-й строки матрицы C
            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   a_val = A.Val[t];

                for (int j = b_pos; j < b_max; ++j)
                    accum.value[B.Col[j]] += a_val * B.Val[j];
            }

            // Заполнение строки матрицы C и очистка аккумулятора
            for (int j = m_min; j < m_max; ++j) {
                C.Val[j] = accum.value[M.Col[j]];
                // accum.value[M.Col[j]] = zero;
            }
        }
    }
}

/*MSpGEMM с использованием кучи*/

/*Итератор для хранения информации о текущем элементе строки матрицы B*/
template <typename T>
struct spm_iterator {
    int b_pos; // позиция элемента столбца матрицы B в массиве
    int b_max_pos;
    int b_col; // индекс элемента столбца матрицы B в массиве
    T   val;   // значение в матрице A, на которое умножается строка

    spm_iterator() {}
    spm_iterator(int x, int y, int z, const T &val):
        b_pos(x), b_max_pos(y), b_col(z), val(val) {}
    spm_iterator(const spm_iterator &it):
        b_pos(it.b_pos), b_max_pos(it.b_max_pos), b_col(it.b_col), val(it.val) {}
};

template <typename T>
bool operator<(const spm_iterator<T> &it1, const spm_iterator<T> it2) {
    return it1.b_col > it2.b_col;
}

template<typename T>
spMtx<T> mxmm_heap(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<T> &M) {
    // инициализация C
    spMtx<T> C(A.m, B.n, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_heap_parallel(A, B, M, C);
    else
        _mxmm_heap_sequential(A, B, M, C);

    return C;
}

template<typename T>
void mxmm_heap(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<T> &M, spMtx<T> &C) {
    // инициализация C
    C.resizeRows(M.m);
    C.resizeVals(M.nz);
    C.n = M.n;
    memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (C.m + 1) * sizeof(int));
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_heap_parallel(A, B, M, C);
    else
        _mxmm_heap_sequential(A, B, M, C);
}

template<typename T>
void _mxmm_heap_parallel(const spMtx<T> &A, const spMtx<T> &B,
                         const spMtx<T> &M, spMtx<T> &C) {

#pragma omp parallel
    {
        int m_pos;      // текущая позиция в маске М
        int m_col;      // текущий столбец в маске M
        int m_max_pos;  // граница для текущей строки маски M
        std::priority_queue<spm_iterator<T>> heap;
        spm_iterator<T> iter;

#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            // заполнение кучи
            // k - позиция начала строки A.Col[j] в массиве матрицы B
            for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j) {
                int k = B.Rst[A.Col[j]];
                heap.emplace(k, B.Rst[A.Col[j]+1], B.Col[k], A.Val[j]);
            }
            m_pos = M.Rst[i];
            m_col = M.Col[m_pos];
            m_max_pos = M.Rst[i+1];

            while (!heap.empty()) {
                // достаём итератор с минимальным индексом столбца B
                iter = heap.top();
                heap.pop();

                // если столбец в М меньше минимального текущего столбца в B,
                // ищем первую не меньшую позицию.
                // если дошли до конца, выходим
                while (m_pos < m_max_pos && m_col < iter.b_col)
                    m_col = M.Col[++m_pos];
                if (m_pos == m_max_pos)
                    break;
                // при совпадении столбцов в M и B умножаем и прибавляем результат
                // и увеличиваем итератор
                if (m_col == iter.b_col && iter.b_pos < iter.b_max_pos)
                    C.Val[m_pos] += iter.val * B.Val[iter.b_pos];
                // обработка вставки итератора обратно в кучу
                // если у итератора столбец B меньше, чем текущий столбец M,
                // увеличиваем, пока он меньше.
                // итератор, дошедший до конца строки в B, не вставляется

                iter.b_col = B.Col[++iter.b_pos];
                while (iter.b_pos < iter.b_max_pos && iter.b_col < m_col)
                    iter.b_col = B.Col[++iter.b_pos];
                if (iter.b_pos < iter.b_max_pos)
                    heap.push(iter);
            }
            heap = std::priority_queue<spm_iterator<T>>();
        }
    }
}

template<typename T>
void _mxmm_heap_sequential(const spMtx<T> &A, const spMtx<T> &B,
                           const spMtx<T> &M, spMtx<T> &C) {
    int m_col;      // текущий столбец в маске M
    int m_pos;      // текущая позиция в маске М
    int m_max_pos;  // граница для текущей строки маски M
    std::priority_queue<spm_iterator<T>> heap;
    spm_iterator<T> iter;

    for (size_t i = 0; i < A.m; ++i) {
            // заполнение кучи
            // k - позиция начала строки A.Col[j] в массиве матрицы B
        for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j) {
            int k = B.Rst[A.Col[j]];
            heap.emplace(k, B.Rst[A.Col[j]+1], B.Col[k], A.Val[j]);
        }
        m_pos = M.Rst[i];
        m_col = M.Col[m_pos];
        m_max_pos = M.Rst[i+1];

        while (!heap.empty()) {
            // достаём итератор с минимальным индексом столбца B
            iter = heap.top();
            heap.pop();

            // если столбец в М меньше минимального текущего столбца в B,
            // ищем первую не меньшую позицию.
            // если дошли до конца, выходим
            while (m_col < iter.b_col && m_pos < m_max_pos)
                m_col = M.Col[++m_pos];
            if (m_pos == m_max_pos)
                break;

            // при совпадении столбцов в M и B умножаем и прибавляем результат
            if (m_col == iter.b_col && iter.b_pos < iter.b_max_pos)
                C.Val[m_pos] += iter.val * B.Val[iter.b_pos];

            // обработка вставки итератора обратно в кучу
            // если у итератора столбец B меньше, чем текущий столбец M,
            // увеличиваем, пока он меньше.
            // итератор, дошедший до конца строки в B, не вставляется

            iter.b_col = B.Col[++iter.b_pos];
            while (iter.b_pos < iter.b_max_pos && iter.b_col < m_col)
                iter.b_col = B.Col[++iter.b_pos];
            if (iter.b_pos < iter.b_max_pos)
                heap.push(iter);
        }
        heap = std::priority_queue<spm_iterator<T>>();
    }
}


template <typename T>
void mxmm_naive(bool isParallel, const spMtx<T> &A, const spMtx<T> &B,
                                 const spMtx<T> &M, spMtx<T> &C) {
    // инициализация C
    C.m = A.m;
    if (!C.Col)
        delete[] C.Col;
    if (!C.Val)
        delete[] C.Val;
    if (!C.Rst)
        delete[] C.Rst;
    C.Rst = new int[A.m + 1];
    C.Rst[0] = 0;

    // Символьная стадия
    if (isParallel == true) {
#pragma omp parallel
        {
            int count;
            char *is_set = new char[A.m]();
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < A.m; ++i) {
                count = 0;
                for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k)
                    for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j)
                        is_set[B.Col[j]] = 1;
                for (int k = 0; k < A.m; ++k)
                    if (is_set[k])
                        ++count;
                memset(is_set, 0, A.m*sizeof(char));
                C.Rst[i+1] = count;
            }
            delete[] is_set;
        }
    } else {
        int count = 0;
        char *is_set = new char[A.m]();
        for (size_t i = 0; i < A.m; ++i) {
            for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k)
                for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j)
                    is_set[B.Col[j]] = 1;
            for (int k = 0; k < A.m; ++k)
                if (is_set[k])
                    ++count;
            C.Rst[i+1] = count;
            memset(is_set, 0, A.m*sizeof(char));
        }
        C.Col = new int[C.Rst[C.m]];
        C.Val = new T[C.Rst[C.m]];
    }

    // Численная стадия
    if (isParallel == true) {
#pragma omp parallel
        {
            T *rowpr = new T[A.m]();
            char *is_set = new char[A.m]();
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < A.m; ++i) {
                int c_curr = C.Rst[i];
                for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
                    for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j) {
                        is_set[B.Col[j]] = 1;
                        rowpr[B.Col[j]] += A.Val[k] * B.Val[j];
                    }
                }
                for (int k = 0; k < A.m; ++k) {
                    if (is_set[k]) {
                        C.Col[c_curr] = k;
                        C.Val[c_curr++] = rowpr[k];
                    }
                }
                memset(is_set, 0, A.m*sizeof(char));
                memset(rowpr, 0, A.m*sizeof(T));
            }
            delete[] rowpr;
            delete[] is_set;
        }
    } else {
        T *rowpr = new T[A.m]();
        char *is_set = new char[A.m]();
        for (size_t i = 0; i < A.m; ++i) {
            int c_curr = C.Rst[i];
            for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
                for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j) {
                    is_set[B.Col[j]] = 1;
                    rowpr[B.Col[j]] += A.Val[k] * B.Val[j];
                }
            }
            for (int k = 0; k < A.m; ++k) {
                if (is_set[k]) {
                    C.Col[c_curr] = k;
                    C.Val[c_curr++] = rowpr[k];
                }
            }
            memset(is_set, 0, A.m*sizeof(char));
            memset(rowpr, 0, A.m*sizeof(T));
        }
        delete[] rowpr;
        delete[] is_set;
    }

    // Применение маски
    T *c_wgt_new = new T[M.nz]();
    int *c_adj_new = new int[M.nz];
    memcpy(c_adj_new, M.Col, M.nz*sizeof(int));
    
    int c_curr;
    if (isParallel == true) {
#pragma omp parallel for private(c_curr) schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            c_curr = C.Rst[i];
            for (int j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
                while (c_curr < C.Rst[i+1] && C.Col[c_curr] < M.Col[j])
                    ++c_curr;
                if (c_curr < C.Rst[i+1] && C.Col[c_curr] == M.Col[j])
                    c_wgt_new[j] = C.Val[c_curr++];
            }
        }
    } else {
        for (size_t i = 0; i < A.m; ++i) {
            c_curr = C.Rst[i];
            for (int j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
                while (c_curr < C.Rst[i+1] && C.Col[c_curr] < M.Col[j])
                    ++c_curr;
                if (c_curr < C.Rst[i+1] && C.Col[c_curr] == M.Col[j])
                    c_wgt_new[j] = C.Val[c_curr++];
            }
        }
    }

    delete[] C.Val;
    delete[] C.Col;
    C.Val = c_wgt_new;
    C.Col = c_adj_new;
    memcpy(C.Rst, M.Rst, (M.m + 1)*sizeof(int));
    C.nz = C.Rst[C.m];

}

template <typename T>
void MxV(const spMtx<T> &G, T *vec, T *res) {
    for (size_t i = 0; i < G.m; ++i)
        for (int j = G.Rst[i]; j < G.Xadj[i+1]; ++j)
            res[i] += G.Val[j] * vec[G.Col[j]];
}

template <typename T>
void VxM(const spMtx<T> &G, T *vec, T *res) {
    for (size_t i = 0; i < G.m; ++i)
        for (int j = G.Rst[i]; j < G.Rst[i+1]; ++j)
            res[G.Col[j]] += G.Val[j] * vec[i];
}

template <typename T>
spMtx<T> transpose(const spMtx<T> &A) {
    spMtx<T> AT(A.n, A.m, A.nz);

    // filling the column indices array and current column positions array
    for (size_t i = 0; i < A.nz; ++i)
        ++AT.Rst[A.Col[i]+1];
    for (size_t i = 0; i < AT.m; ++i)
        AT.Rst[i+1] += AT.Rst[i];

    // transposing
    for (size_t i = 0; i < A.m; ++i) {
        for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j) {
            AT.Val[AT.Rst[A.Col[j]]] = std::move(A.Val[j]);
            AT.Col[AT.Rst[A.Col[j]]++] = i;
        }
    }
    // set Rst indices to normal state
    // AT.Rst[AT.m] already has the correct value
    for (int i = AT.m - 1; i > 0; --i)
        AT.Rst[i] = AT.Rst[i-1];
    AT.Rst[0] = 0;

    return AT;
}


std::vector<float> betweenness_centrality(bool isParallel, const spMtx<int> &A, size_t blockSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
    size_t mxn;
    std::vector<float> bcv(A.m);
    float *bc = bcv.data();
    std::vector<spMtx<int>> Sigmas(A.m);
    spMtx<int> AT = transpose(A);
    spMtx<int> Front;
    spMtx<int> Fronttmp;
    spMtx<int> Numsp(m, 0);
    spMtx<float> Af = convertType<float>(A);
    denseMtx<float> Numspd;
    denseMtx<float> Bcu;
    denseMtx<float> Nspinv;
    denseMtx<float> W;
    denseMtx<float> Wtmp;

    for (size_t i = 0; i < A.m; i += blockSize) {
        size_t n = std::min(A.n - i, blockSize); // количество столбцов
        size_t mxn = (size_t)m * n;

        Numsp.resizeVals(n);
        Numsp.n = n;
        for (size_t j = 0; j < i; ++j)
            Numsp.Rst[j] = 0;
        for (size_t j = 0; j < n; ++j) {
            Numsp.Rst[i+j] = j;
            Numsp.Col[j] = j;
            Numsp.Val[j] = 1;
        }
        for (size_t j = i+n; j <= m; ++j)
            Numsp.Rst[j] = n;
        Front = transpose(A.extractRows(i, i+n));

        // Прямой проход (поиск в ширину)
        size_t d = 0;
        do {
            Sigmas[d] = Front;
            Numsp = eWiseAdd(Numsp, Front);
            mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
            Front = Fronttmp;
            ++d;
        } while (Front.nz);


        // Обратный проход
        Numspd = Numsp;
        Nspinv.resize(m, n);
        Bcu.resize(m, n);
        W.resize(m, n);
        for (size_t i = 0; i < mxn; ++i)
            Bcu.Val[i] = 1.0f;
        for (size_t i = 0; i < mxn; ++i)
            Nspinv.Val[i] = 1.0f / Numspd.Val[i];

        for (size_t k = d-1; k > 0; --k) {
            eWiseMult(Nspinv, Bcu, Sigmas[k], W);
            mxmm(Af, W, Sigmas[k-1], W);
            fuseEWiseMultAdd(W, Numspd, Bcu);
        }

        // Редукция посчитанных значений в 'bc'
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < m; ++i) {
            float *bcu_ptr = Bcu.Val + (size_t)n*i;
            for (size_t j = 0; j < n; ++j) {
                bc[i] += bcu_ptr[j];
            }
            bc[i] -= (float)n;
        }

        if (i + n >= m/4)
            std::cerr << "Done 25%\n";
        else if (i + n >= m/2)
            std::cerr << "Done 50%\n";
        else if (i + n >= 3*m/4)
            std::cerr << "Done 75%\n";
        else if (i + n == m)
            std::cerr << "Done 100%\n";
    }

    return bcv;
}

#endif
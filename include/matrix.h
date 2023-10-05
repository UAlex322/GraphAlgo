#pragma once

#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
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

    void resizeVals(size_t newNz) {
        if (newNz > capacity) {
            if (Col != nullptr)
                delete[] Col;
            if (Val != nullptr)
                delete[] Val;
            Col = new  int[newNz];
            Val = new ValT[newNz];
            capacity = newNz;
        }
        nz = newNz;
    }

    void resizeRows(size_t newM) {
        if (m != newM) {
            if (Rst != nullptr)
                delete[] Rst;
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
                Col[last_el[row_a[i]]++] = col_a[i];
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

        if (fread(matcode, 1, 1, fp) != 1)          return -2;
        if (fread(matcode + 1, 1, 1, fp) != 1)      return -2;
        if (fread(matcode + 2, 1, 1, fp) != 1)      return -2;
        if (fread(matcode + 3, 1, 1, fp) != 1)      return -2;
        if (fread(&m, sizeof(size_t), 1, fp) != 1)  return -2;
        if (fread(&n, sizeof(size_t), 1, fp) != 1)  return -2;
        if (fread(&nz, sizeof(size_t), 1, fp) != 1) return -2;


        Rst = new int[m+1];
        Col = new int[nz];
        Val = new ValT[nz];
        int  *RstTmp = new int[m+1];
        int  *ColTmp = new int[nz];
        ValT *ValTmp = new ValT[nz];
        if (fread(RstTmp, sizeof(int), m+1, fp) != m+1) return -3;
        if (fread(ColTmp, sizeof(int), nz, fp)  != nz ) return -3;
        if (fread(ValTmp, sizeof(ValT), nz, fp) != nz ) return -3;

        // write with parallel for into main arrays to get along with NUMA
    #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            Rst[i] = RstTmp[i];
            Rst[i+1] = RstTmp[i+1];
        }
    #pragma omp parallel for
        for (size_t i = 0; i < nz; ++i) {
            Col[i] = ColTmp[i];
            Val[i] = ValTmp[i];
        }

        fclose(fp);
        delete[] RstTmp;
        delete[] ColTmp;
        delete[] ValTmp;
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

template <typename ValT2, typename ValT1>
spMtx<ValT2> convertType(const spMtx<ValT1> &source) {
    spMtx<ValT2> result;

    result.copyPattern(source);
    for (size_t j = 0; j < source.nz; ++j)
        result.Val[j] = (ValT2)(source.Val[j]);

    return result;
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
    Res.n = Gr.n;
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
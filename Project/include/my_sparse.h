#ifndef MY_SPARSE_H 
#define MY_SPARSE_H

#include "spMatrix.h"
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <random>
#include <set>
#include <queue>
#include <chrono>
#include <omp.h>

spMatrix<int> generate_mask(const int n, const int min_deg, const int max_deg) {
    std::set<int> positions;
    std::uniform_int_distribution<int> deg_distr(min_deg, max_deg);
    std::uniform_int_distribution<int> col_distr(0, n - 1);
    std::mt19937 generator{std::random_device{}()};
    spMatrix<int> mask;
    mask.Rst = new int[n+1];
    mask.m = mask.n = n;

    mask.Rst[0] = 0;
    for (int i = 0; i < n; ++i)
        mask.Rst[i+1] = mask.Rst[i] + deg_distr(generator);
    mask.nz = mask.Rst[n];
    mask.Col = new int[mask.nz];

    int j = 0;
    for (int i = 0; i < mask.n; ++i) {
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
void full_mask(spMatrix<T> &mask, const int n) {
    mask.v = n;
    mask.nz = n*n;
    mask.Rst = new int[n+1];
    mask.Col = new int[(int64_t)n*n];

    for (int i = 0; i <= n; ++i)
        mask.Rst[i] = i*n;
    int64_t curr_pos = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            mask.Col[curr_pos++] = j;
    }
}

template <typename T>
spMatrix<int> build_adjacency_matrix(const spMatrix<T> &Gr) {
    spMatrix<int> Res;
    Res.v = Gr.v;
    Res.nz = Gr.nz;
    Res.Col = new int[Gr.nz];
    Res.Rst = new int[Gr.v + 1];
    Res.Val = new int[Gr.nz];

    std::memcpy(Res.Col, Gr.Col, Gr.nz*sizeof(int));
    std::memcpy(Res.Rst, Gr.Rst, (Gr.v + 1)*sizeof(int));
    std::memcpy(Res.matcode, Gr.matcode, 4*sizeof(char));
    Res.matcode[2] = 'I';

    for (int i = 0; i < Gr.nz; ++i)
        Res.Val[i] = 1;

    return Res;
}

template <typename T>
spMatrix<T> build_symm_from_lower(const spMatrix<T> &Low) {
    spMatrix<T> Res(Low.v, 2*Low.nz);
    spMatrix<T> Upp = transpose(Low);
    int jl = 0;
    int ju = 0;
    int jr = 0;

    for (int i = 0; i < Low.v; ++i) {
        int xl = Low.Rst[i+1];
        int xu = Upp.Rst[i+1];

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
spMatrix<T> extract_lower_triangle(const spMatrix<T> &Gr) {
    spMatrix<T> Res;

    Res.v = Gr.v;
    Res.Rst = new int[Gr.v + 1];
    Res.Rst[0] = 0;

    for (int i = 0; i < Gr.v; ++i) {
        int r = Gr.Rst[i];
        while (r < Gr.Rst[i+1] && Gr.Col[r] < i)
            ++r;
        Res.Rst[i+1] = Res.Rst[i] + (r - Gr.Rst[i]);
    }

    Res.nz = Res.Rst[Res.v];
    Res.Col = new int[Res.nz];
    Res.Val = new   T[Res.nz];

    for (int i = 0; i < Gr.v; ++i) {
        int row_len = Res.Rst[i+1] - Res.Rst[i];
        std::memcpy(Res.Col + Res.Rst[i], Gr.Col + Gr.Rst[i], row_len*sizeof(int));
        std::memcpy(Res.Val + Res.Rst[i], Gr.Val + Gr.Rst[i], row_len*sizeof(T));
    }
    
    return Res;
}

template <typename T>
spMatrix<T> extract_upper_triangle(const spMatrix<T> &Gr) {
    spMatrix<T> Res;

    Res.v = Gr.v;
    Res.Rst = new int[Gr.v + 1];
    Res.Rst[0] = 0;

    for (int i = 0; i < Gr.v; ++i) {
        int r = Gr.Rst[i];
        while (r < Gr.Rst[i+1] && Gr.Col[r] <= i)
            ++r;
        Res.Rst[i+1] = Res.Rst[i] + (Gr.Rst[i+1] - r);
    }

    Res.nz = Res.Rst[Res.v];
    Res.Col = new int[Res.nz];
    Res.Val = new   T[Res.nz];

    for (int i = 0; i < Gr.v; ++i) {
        int row_len = Res.Rst[i+1] - Res.Rst[i];
        int row_offset = Gr.Rst[i+1] - row_len;
        std::memcpy(Res.Col + Res.Rst[i], Gr.Col + row_offset, row_len*sizeof(int));
        std::memcpy(Res.Val + Res.Rst[i], Gr.Val + row_offset, row_len*sizeof(T));
    }

    return Res;
}

spMatrix<int> generate_adjacency_matrix(const int n, const int min_deg, const int max_deg) {
    spMatrix<int> Res = generate_mask(n, min_deg, max_deg);

    Res.Val = new int[Res.nz];
    for (int j = 0; j < Res.nz; ++j)
        Res.Val[j] = 1;

    return build_symm_from_lower(extract_lower_triangle(Res));
}

template <typename T>
spMatrix<T> add(const spMatrix<T> &A, const spMatrix<T> &B) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMatrix<T> C(A.m, 0);
    for (int i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int colCnt = 0;
        while (aIdx < A.Rst[i+1] && bIdx < B.Rst[i+1]) {
            if (A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            else if (A.Col[bIdx] > B.Col[bIdx])
                ++bIdx;
            else {
                ++aIdx;
                ++bIdx;
            }
            ++colCnt;
        }
        C.Rst[i+1] = colCnt;
    }

    for (int i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.nz = C.Rst[C.m];
    C.Val = new T[C.nz];
    
    for (int i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        for (int j = C.Rst[i]; j < C.Rst[i+1]; ++j) {
            if (A.Col[aIdx] < B.Col[bIdx]) {
                C.Col[j] = A.Col[aIdx];
                C.Val[j] = A.Val[aIdx++];
            }
            else if (A.Col[bIdx] > B.Col[bIdx]) {
                C.Col[j] = B.Col[bIdx];
                C.Val[j] = B.Val[bIdx++];
            }
            else {
                C.Col[j] = A.Col[aIdx];
                C.Val[j] = A.Val[aIdx++] + B.Val[bIdx++];
            }
        }
    }

    return C;
}

template <typename MatrixValT, typename ScalarT>
spMatrix<MatrixValT> multScalar(const spMatrix<MatrixValT> &A, const ScalarT &alpha) {
    MatrixValT *valEnd  = A.Val + A.nz;
    for (MatrixValT *valCurr = A.Val; valCurr < valEnd; ++valCurr)
        *valCurr *= alpha;
    return A;
}

template <typename int>
double *betweenness_centrality(const spMatrix<int> &G) {
    double *bc       = new double[G.v]();
    double *delta    = new double[G.v];
    uint64_t *pathCnt = new uint64_t[G.v];
    bool *wasnt = new bool[G.v];
    std::queue<int> bfs_queue;
    // using PairType = std::pair<WgtType, int>;
    // std::priority_queue<PairType, std::vector<PairType>, std::greater<PairType>> queue;

    for (int i = 0; i < G.v; ++i) {
        for (int i = 0; i < G.v; ++i)
            wasnt[i] = true;
        std::memset(pathCnt, 0, G.v * sizeof(uint64_t));
        wasnt[i] = false;
        pathCnt[i] = 1;
        bfs_queue.push(i);

        while (!bfs_queue.empty()) {
            int currVertex = bfs_queue.front();
            bfs_queue.pop();

            for (int j = G.Rst[currVertex]; j < G.Rst[currVertex + 1]; ++j)
                if (wasnt[G.Col[j]])
                    pathCnt[G.Col[j]] += pathCnt[currVertex];
            for (int j = G.Rst[currVertex]; j < G.Rst[currVertex + 1]; ++j)
                if (wasnt[G.Col[j]]) {
                    wasnt[G.Col[j]] = false;
                    bfs_queue.push
                }
        }
    }

    delete delta;
    delete wasnt;
    return bc;
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

template<typename T>
spMatrix<T> mxmm_mca(const spMatrix<T> &A, const spMatrix<T> &B, const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C(M.v, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.v + 1) * sizeof(int));

    // Вычисление максимальной длины строки маски,
    // чтобы единожды зарезервировать достаточно памяти
    int max_len = 0;
    for (int i = 0; i < A.v; ++i)
        if (M.Rst[i+1] - M.Rst[i] > max_len)
            max_len = M.Rst[i+1] - M.Rst[i];

    MCA<T> accum(max_len);

    for (int i = 0; i < A.v; ++i) {
        int m_row_len  = M.Rst[i+1] - M.Rst[i];
        int m_pos;

        // подсчёт i-й строки матрицы C
        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   A_val = A.Val[t];
            // прохождение строки и обработка только входящих в маску элементов
            m_pos = M.Rst[i];
            for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                // ищем следующий входящий в маску элемент
                while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                    ++b_pos;
                // при нахождении накапливаем значение
                if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                    accum.values[j] += A_val * B.Val[b_pos];
            }
        }
        // заполнение i-й строки матрицы C
        memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
        // очистка аккумулятора для следующей итерации
        memset(accum.values, 0, max_len * sizeof(T));
    }

    return C;
}

template<typename T>
spMatrix<T> mxmm_mca_par(const spMatrix<T> &A, const spMatrix<T> &B, const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C(M.v, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.v + 1) * sizeof(int));

    // Вычисление максимальной длины строки маски,
    // чтобы единожды зарезервировать достаточно памяти
    int max_len = 0;
    for (int i = 0; i < A.v; ++i)
        if (M.Rst[i+1] - M.Rst[i] > max_len)
            max_len = M.Rst[i+1] - M.Rst[i];

#pragma omp parallel
    {
        MCA<T> accum(max_len);

    #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
            int m_row_len  = M.Rst[i+1] - M.Rst[i];
            int m_pos;

            // подсчёт i-й строки матрицы C
            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k+1];
                T   A_val = A.Val[t];
                // прохождение строки и обработка только входящих в маску элементов
                m_pos = M.Rst[i];
                for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                    // ищем следующий входящий в маску элемент
                    while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                        ++b_pos;
                    // при нахождении накапливаем значение
                    if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                        accum.values[j] += A_val * B.Val[b_pos];
                }
            }
            
            // заполнение i-й строки матрицы C
            memcpy(C.Val + C.Rst[i], accum.values, m_row_len*sizeof(T));
            // очистка аккумулятора для следующей итерации
            accum.clear();
        }
    }

    return C;
}

template<typename T>
spMatrix<T> mxmcm_mca(const spMatrix<T> &A, const spMatrix<T> &B, const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C(M.v, 0);

    // длина аккумулятора, при которой все строки поместятся
    int accum_len = 0;

#pragma omp parallel
    {
        MCA<char> accum(A.v);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
            int m_pos = M.Rst[i];
            int m_max = M.Rst[i + 1];
            int c_row_len = 0;

            for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k + 1];
                for (int j = b_pos; j < b_max; ++j)
                    accum[B.Col[j]] = SET;
            }
            for (int j = m_pos; j < m_max; ++j)
                accum[M.Col[j]] = ALLOWED;

            for (int j = 0; j < A.v; ++j)
                c_row_len += accum[j];
            C.Rst[i+1] = c_row_len;
            if (c_row_len > accum_len)
                accum_len = c_row_len;
        }
    }

    for (int i = 0; i < C.v; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.nz = C.Rst[C.v];
    C.Val = new T[C.nz];
    C.Col = new int[C.nz];

    // численная стадия
#pragma omp parallel 
    {
        MCA<T> accum(accum_len);
#pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
            int m_row_len = M.Rst[i + 1] - M.Rst[i];
            int m_pos;

            // подсчёт i-й строки матрицы C
            for (int t = A.Rst[i]; t < A.Rst[i + 1]; ++t) {
                int k = A.Col[t];
                int b_pos = B.Rst[k];
                int b_max = B.Rst[k + 1];
                T   A_val = A.Val[t];
                // прохождение строки и обработка только не входящих в маску элементов
                m_pos = M.Rst[i];
                for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                    // ищем следующий входящий в маску элемент
                    while (b_pos < b_max && B.Col[b_pos] < M.Col[m_pos])
                        ++b_pos;
                    // при нахождении накапливаем значение
                    if (b_pos < b_max && B.Col[b_pos] == M.Col[m_pos])
                        accum.values[j] += A_val * B.Val[b_pos];
                }
            }

            // заполнение i-й строки матрицы C
            memcpy(C.Val + C.Rst[i], accum.values, m_row_len * sizeof(T));
            accum.clear();
        }
    }

    return C;
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
spMatrix<T> mxmm_heap(const spMatrix<T> &A, const spMatrix<T> &B,
                       const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C(M.v, M.nz);
    memcpy(C.Col, M.Col, C.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.v + 1)*sizeof(int));

    int m_col;      // текущий столбец в маске M
    int m_pos;      // текущая позиция в маске М
    int m_max_pos;  // граница для текущей строки маски M
    std::priority_queue<spm_iterator<T>> heap;
    spm_iterator<T> iter;

    for (int i = 0; i < A.v; ++i) {
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

    return C;
}



template<typename T>
spMatrix<T> mspgemm_heap_parallel(const spMatrix<T> &A, const spMatrix<T> &B,
                                   const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C(M.v, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.v + 1) * sizeof(int));

#pragma omp parallel
    {
        int m_pos;      // текущая позиция в маске М
        int m_col;      // текущий столбец в маске M
        int m_max_pos;  // граница для текущей строки маски M
        std::priority_queue<spm_iterator<T>> heap;
        spm_iterator<T> iter;

    #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
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

    return C;
}

template <typename T>
spMatrix<T> mspgemm_naive(const spMatrix<T> &A, const spMatrix<T> &B,
                           const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C;
    C.v = A.v;
    if (!C.Col)
        delete[] C.Col;
    if (!C.Val)
        delete[] C.Val;
    if (!C.Rst)
        delete[] C.Rst;
    C.Rst = new int[A.v + 1];
    C.Rst[0] = 0;

    // Символьная стадия
    int count = 0;
    char *is_set = new char[A.v]();
    for (int i = 0; i < A.v; ++i) {
        for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k)
            for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j)
                is_set[B.Col[j]] = 1;
        for (int k = 0; k < A.v; ++k)
            if (is_set[k])
                ++count;
        C.Rst[i+1] = count;
        memset(is_set, 0, A.v*sizeof(char));
    }
    C.Col = new int[C.Rst[C.v]];
    C.Val = new T[C.Rst[C.v]];

    // Численная стадия

    T *rowpr = new T[A.v]();
    for (int i = 0; i < A.v; ++i) {
        int c_curr = C.Rst[i];
        for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
            for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j) {
                is_set[B.Col[j]] = 1;
                rowpr[B.Col[j]] += A.Val[k] * B.Val[j];
            }
        }
        for (int k = 0; k < A.v; ++k) {
            if (is_set[k]) {
                C.Col[c_curr] = k;
                C.Val[c_curr++] = rowpr[k];
            }
        }
        memset(is_set, 0, A.v*sizeof(char));
        memset(rowpr, 0, A.v*sizeof(T));
    }
    delete[] is_set;
    delete[] rowpr;

    // Применение маски
    T *c_wgt_new = new T[M.nz]();
    int *c_adj_new = new int[M.nz];
    memcpy(c_adj_new, M.Col, M.nz*sizeof(int));
    
    int c_curr;
    for (int i = 0; i < A.v; ++i) {
        c_curr = C.Rst[i];
        for (int j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
            while (c_curr < C.Rst[i+1] && C.Col[c_curr] < M.Col[j])
                ++c_curr;
            if (c_curr < C.Rst[i+1] && C.Col[c_curr] == M.Col[j])
                c_wgt_new[j] = C.Val[c_curr++];
        }
    }
    delete[] C.Val;
    delete[] C.Col;
    C.Val = c_wgt_new;
    C.Col = c_adj_new;
    memcpy(C.Rst, M.Rst, (M.v + 1)*sizeof(int));
    C.nz = C.Rst[C.v];

    return C;
}

template <typename T>
spMatrix<T> mspgemm_naive_parallel(const spMatrix<T> &A, const spMatrix<T> &B, const spMatrix<T> &M) {
    // инициализация C
    spMatrix<T> C;
    C.v = A.v;
    if (!C.Col)
        delete[] C.Col;
    if (!C.Val)
        delete[] C.Val;
    if (!C.Rst)
        delete[] C.Rst;
    C.Rst = new int[A.v + 1];
    C.Rst[0] = 0;

    // Символьная стадия
#pragma omp parallel
    {
        int count;
        char *is_set = new char[A.v]();
    #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
            count = 0;
            for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k)
                for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j)
                    is_set[B.Col[j]] = 1;
            for (int k = 0; k < A.v; ++k)
                if (is_set[k])
                    ++count;
            memset(is_set, 0, A.v*sizeof(char));
            C.Rst[i+1] = count;
        }
        delete[] is_set;
    }
    for (int i = 0; i < A.v; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.Col = new int[C.Rst[C.v]];
    C.Val = new T[C.Rst[C.v]];

    // Численная стадия
#pragma omp parallel
    {
        T *rowpr = new T[A.v]();
        char *is_set = new char[A.v]();
    #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.v; ++i) {
            int c_curr = C.Rst[i];
            for (int k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
                for (int j = B.Rst[A.Col[k]]; j < B.Rst[A.Col[k] + 1]; ++j) {
                    is_set[B.Col[j]] = 1;
                    rowpr[B.Col[j]] += A.Val[k] * B.Val[j];
                }
            }
            for (int k = 0; k < A.v; ++k) {
                if (is_set[k]) {
                    C.Col[c_curr] = k;
                    C.Val[c_curr++] = rowpr[k];
                }
            }
            memset(is_set, 0, A.v*sizeof(char));
            memset(rowpr, 0, A.v*sizeof(T));
        }
        delete[] rowpr;
        delete[] is_set;
    }
    // Применение маски
    T *c_wgt_new = new T[M.nz]();
    int *c_adj_new = new int[M.nz];
    memcpy(c_adj_new, M.Col, M.nz*sizeof(int));

    int c_curr;
#pragma omp parallel for private(c_curr) schedule(dynamic)
    for (int i = 0; i < A.v; ++i) {
        c_curr = C.Rst[i];
        for (int j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
            while (c_curr < C.Rst[i+1] && C.Col[c_curr] < M.Col[j])
                ++c_curr;
            if (c_curr < C.Rst[i+1] && C.Col[c_curr] == M.Col[j])
                c_wgt_new[j] = C.Val[c_curr++];
        }
    }
    delete[] C.Val;
    delete[] C.Col;
    C.Val = c_wgt_new;
    C.Col = c_adj_new;
    memcpy(C.Rst, M.Rst, (M.v + 1)*sizeof(int));
    C.nz = C.Rst[C.v];

    return C;
}

template <typename T>
void MxV(const spMatrix<T> &G, T *vec, T *res) {
    for (int i = 0; i < G.v; ++i)
        for (int j = G.Rst[i]; j < G.Xadj[i+1]; ++j)
            res[i] += G.Val[j] * vec[G.Col[j]];
}

template <typename T>
void VxM(const spMatrix<T> &G, T *vec, T *res) {
    for (int i = 0; i < G.v; ++i)
        for (int j = G.Rst[i]; j < G.Rst[i+1]; ++j)
            res[G.Col[j]] += G.Val[j] * vec[i];
}

template <typename T>
spMatrix<T> transpose(const spMatrix<T> &G) {
    spMatrix<T> NG(G.v, G.nz);

    // filling the column indices array and current column positions array
    for (int i = 0; i < G.nz; ++i)
        ++NG.Rst[G.Col[i]+1];
    for (int i = 0; i < G.v; ++i)
        NG.Rst[i+1] += NG.Rst[i];

    // transposing
    for (int i = 0; i < G.v; ++i) {
        for (int j = G.Rst[i]; j < G.Rst[i+1]; ++j) {
            NG.Val[NG.Rst[G.Col[j]]] = std::move(G.Val[j]);
            NG.Col[NG.Rst[G.Col[j]]++] = i;
        }
    }
    // set Rst indices to normal state
    // NG.Rst[NG.v] already has the correct value
    for (int i = G.v - 1; i > 0; --i)
        NG.Rst[i] = NG.Rst[i-1];
    NG.Rst[0] = 0;

    return NG;
}

/*
sparse_matrix_t create_mkl_spm(const spMatrix<double> &gr) {
int status;
sparse_matrix_t mkl_spm;
if (status = mkl_sparse_d_create_csr(&mkl_spm, SPARSE_INDEX_BASE_ZERO, gr.v, gr.v, gr.Rst, gr.Rst + 1, gr.Col, gr.Val))
exit(status);

return mkl_spm;
}

spMatrix<double> mkl_mspmm(const spMatrix<double> &A, const spMatrix<double> &B, const spMatrix<double> &M) {
int n = A.v;
int status;
sparse_matrix_t mkl_sp_a = create_mkl_spm(A);
sparse_matrix_t mkl_sp_b = create_mkl_spm(B);
sparse_matrix_t mkl_product;

mkl_sparse_optimize(mkl_sp_a);
mkl_sparse_optimize(mkl_sp_b);

chrono::steady_clock::time_point mkl_start = chrono::steady_clock::now();
if (status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mkl_sp_a, mkl_sp_b, &mkl_product))
exit(status);
chrono::steady_clock::time_point mkl_finish = chrono::steady_clock::now();
printf("MKL  time: %ld ms\n", chrono::duration_cast<chrono::milliseconds>(mkl_finish - mkl_start).count());

spMatrix<double> C;
sparse_index_base_t indexing;
mkl_sparse_d_export_csr(mkl_product, &indexing, &C.v, &C.v, &C.Rst, &C.Rst + 1, &C.Col, &C.Val);
C.nz = C.Rst[n];

int c_last = 0;
int c_curr = 0;
for (int i = 0; i < C.v; ++i) {
for (int j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
while (c_curr < C.Rst[i] && C.Col[c_curr] < M.Col[j])
++c_curr;
if (c_curr < C.Rst[i+1] && C.Col[c_curr] == M.Col[j]) {
C.Col[c_last] = C.Col[c_curr];
C.Val[c_last++] = C.Val[c_curr++];
}
}
C.Rst[i+1] = c_last;
}


//mkl_sparse_destroy(mkl_sp_a);
//mkl_sparse_destroy(mkl_sp_b);
//mkl_sparse_destroy(mkl_product);
return C;
}
*/

#endif
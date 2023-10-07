#pragma once

#include "matrix.h"
#include <queue>
#include <vector>
#include <utility>

// declarations

template <typename T>
spMtx<T> transpose(const spMtx<T> &A);

template <typename T>
void fuseEWiseMultAdd(const denseMtx<T> &A,
                      const denseMtx<T> &B,
                      const denseMtx<T> &C);

template <typename T, typename U> 
void eWiseMult(const denseMtx<T> &A,
               const denseMtx<T> &B,
               const spMtx<U> &M,
               const denseMtx<T> &C);

template <typename T> denseMtx<T> transpose(const denseMtx<T> &A);

template <typename T, typename U>
void mxmm_spd(const spMtx<T> &A,
              const denseMtx<T> &B,
              const spMtx<U> &M,
              denseMtx<T> &C);

template <typename T, typename U> 
spMtx<T> eWiseAdd(const spMtx<T> &A,                  
                  const spMtx<T> &B,
                  const spMtx<U> &M);

template <typename T>
spMtx<T> add_nointersect(const spMtx<T> &A,
                         const spMtx<T> &B);

template <typename T>
spMtx<T> eWiseAdd(const spMtx<T> &A,
                  const spMtx<T> &B);

template <typename T>
spMtx<T> eWiseMult(const spMtx<T> &A,
                   const spMtx<T> &B);

template <typename T, typename U>
spMtx<T> eWiseMult(const spMtx<T> &A,
                   const spMtx<T> &B,
                   const spMtx<U> &M);

template <typename MatrixValT, typename ScalarT>
spMtx<MatrixValT> multScalar(const spMtx<MatrixValT> &A,
                             const ScalarT &alpha);

template<typename T, typename U>
spMtx<T> mxmm_mca(bool isParallel,
                  const spMtx<T> &A,
                  const spMtx<U> &B,
                  const spMtx<T> &M);

template<typename T, typename U>
void mxmm_mca(bool isParallel,
              const spMtx<T> &A,
              const spMtx<T> &B,
              const spMtx<U> &M,
              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_mca_parallel(const spMtx<T> &A,
                        const spMtx<T> &B,
                        const spMtx<U> &M,
                        spMtx<T> &C);

template<typename T, typename U>
void _mxmm_mca_sequential(const spMtx<T> &A,
                          const spMtx<T> &B,
                          const spMtx<U> &M,
                          spMtx<T> &C);

template<typename T, typename U>
void mxmm_msa(bool isParallel,
              const spMtx<T> &A,
              const spMtx<T> &B,
              const spMtx<U> &M,
              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_parallel(const spMtx<T> &A,
                        const spMtx<T> &B,
                        const spMtx<U> &M,
                        spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_sequential(const spMtx<T> &A,
                          const spMtx<T> &B,
                          const spMtx<U> &M,
                          spMtx<T> &C);

template<typename T, typename U>
void mxmm_msa_cmask(bool isParallel,
                    const spMtx<T> &A,
                    const spMtx<T> &B,
                    const spMtx<U> &M,
                    spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_cmask_parallel(const spMtx<T> &A,
                              const spMtx<T> &B,
                              const spMtx<U> &M,
                              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_cmask_sequential(const spMtx<T> &A,
                                const spMtx<T> &B,
                                const spMtx<U> &M,
                                spMtx<T> &C);

template<typename T>
spMtx<T> mxmm_heap(bool isParallel,
                   const spMtx<T> &A,
                   const spMtx<T> &B,
                   const spMtx<T> &M);

template<typename T>
void mxmm_heap(bool isParallel,
               const spMtx<T> &A,
               const spMtx<T> &B,
               const spMtx<T> &M,
               spMtx<T> &C);

template<typename T>
void _mxmm_heap_parallel(const spMtx<T> &A,
                         const spMtx<T> &B,
                         const spMtx<T> &M,
                         spMtx<T> &C);

template<typename T>
void _mxmm_heap_sequential(const spMtx<T> &A,
                           const spMtx<T> &B,
                           const spMtx<T> &M,
                           spMtx<T> &C);

template <typename T>
void mxmm_naive(bool isParallel,
                const spMtx<T> &A,
                const spMtx<T> &B,
                const spMtx<T> &M,
                spMtx<T> &C);

template <typename T>
void MxV(const spMtx<T> &G,
         T *vec,
         T *res);

template <typename T>
void VxM(const spMtx<T> &G,
         T *vec,
         T *res);


// definitions

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

// C += A .* B
template <typename T>
void fuseEWiseMultAdd(const denseMtx<T> &A, const denseMtx<T> &B, denseMtx<T> &C) {
#pragma omp parallel for simd schedule(static, 4096)
    for (size_t i = 0; i < A.m * A.n; ++i)
        C.Val[i] += A.Val[i] * B.Val[i];
}

// C<M> = A .* B
template <typename T, typename U>
void eWiseMult(const denseMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, const denseMtx<T> &C) {
    const T zero = T(0);
#pragma omp parallel for simd schedule(static, 64)
    for (size_t i = 0; i < C.m; ++i) {
        T *c_row = C.Val + i * C.n;
        for (size_t j = 0; j < C.n; ++j) {
            *c_row = zero;
            ++c_row;
        }
    }
#pragma omp parallel for schedule(dynamic, 64)
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
// TODO ^T !!!!!!!!!!!!!!!!
template <typename T, typename U>
void mxmm_spd(const spMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, denseMtx<T> &C, denseMtx<T> &Cbuf) {
    const T zero = T(0);
    // denseMtx<T> BT = transpose(B);

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < M.m; ++i) {
        // for (size_t q = M.Rst[i]; q < M.Rst[i+1]; ++q) {
        //     size_t j = M.Col[q];
        //     T *b_row = BT.Val + BT.n * j;
        //     T dotpr = zero;
        //     for (size_t k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
        //         dotpr += A.Val[k] * b_row[A.Col[k]];
        //     }
        //     Ccopy.Val[C.n * i + j] = dotpr;
        // }
        T *c_row = Cbuf.Val + Cbuf.n * i;
        for (size_t i = 0; i < C.n; ++i)
            c_row[i] = zero;
        for (size_t q = A.Rst[i]; q < A.Rst[i+1]; ++q) {
            size_t j = A.Col[q];
            T  a_val = A.Val[q];
            T *b_row = B.Val + B.n * j;
        #pragma omp simd
            for (size_t k = M.Rst[i]; k < M.Rst[i+1]; ++k)
                c_row[M.Col[k]] += a_val * b_row[M.Col[k]];
        }
    }
    std::swap(C.Val, Cbuf.Val);
}

template <typename T, typename U>
spMtx<T> eWiseAdd(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);
    const T zero = (T)0;

#pragma omp parallel 
    {
        T *row = new T[A.n];

#pragma omp for schedule(dynamic)
        for (int i = 0; i < A.m; ++i) {
            int m_pos = M.Rst[i], m_max = M.Rst[i+1];
            int col_cnt = 0;

            for (int j = m_pos; j < m_max; ++j)
                row[M.Col[j]] = zero;
            for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j)
                row[A.Col[j]] += A.Val[j];
            for (int j = B.Rst[i]; j < B.Rst[i+1]; ++j)
                row[A.Col[j]] += B.Val[j];
            for (int j = m_pos; j < m_max; ++j)
                if (row[M.Col[j]])
                    ++col_cnt;

            C.Rst[i+1] = col_cnt;
        }

        delete[] row;
    }

    for (int i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.resizeVals(C.Rst[A.m]);
    
#pragma omp parallel 
    {
        T *row = new T[A.n];

#pragma omp for schedule(dynamic)
        for (int i = 0; i < A.m; ++i) {
            int m_pos = M.Rst[i], m_max = M.Rst[i+1];
            int c_pos = C.Rst[i];

            for (int j = m_pos; j < m_max; ++j)
                row[M.Col[j]] = zero;
            for (int j = A.Rst[i]; j < A.Rst[i+1]; ++j)
                row[A.Col[j]] += A.Val[j];
            for (int j = B.Rst[i]; j < B.Rst[i+1]; ++j)
                row[A.Col[j]] += B.Val[j];
            for (int j = m_pos; j < m_max; ++j)
                if (row[M.Col[j]]) {
                    C.Col[c_pos] = M.Col[j];
                    C.Val[c_pos++] = row[M.Col[j]];
                }
        }

        delete[] row;
    }

    return C;
}

template <typename T>
spMtx<T> add_nointersect(const spMtx<T> &A, const spMtx<T> &B) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);
    for (size_t i = 1; i <= A.m; ++i)
        C.Rst[i] = A.Rst[i] + B.Rst[i];
    C.resizeVals(C.Rst[C.m]);

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i], cIdx;
        for (cIdx = C.Rst[i]; aIdx < A.Rst[i+1] && bIdx < B.Rst[i+1]; ++cIdx) {
            if (A.Col[aIdx] < B.Col[bIdx]) {
                C.Col[cIdx] = A.Col[aIdx];
                C.Val[cIdx] = A.Val[aIdx++];
            } else {
                C.Col[cIdx] = B.Col[bIdx];
                C.Val[cIdx] = B.Val[bIdx++];
            }
        }
        if (aIdx < A.Rst[i+1]) {
            memcpy(C.Col + cIdx, A.Col + aIdx, (C.Rst[i+1] - cIdx) * sizeof(int));
            memcpy(C.Val + cIdx, A.Val + aIdx, (C.Rst[i+1] - cIdx) * sizeof(T));
        } else {
            memcpy(C.Col + cIdx, B.Col + bIdx, (C.Rst[i+1] - cIdx) * sizeof(int));
            memcpy(C.Val + cIdx, B.Val + bIdx, (C.Rst[i+1] - cIdx) * sizeof(T));
        }
    }

    for (int i = 0; i < C.m; ++i)
        for (int j = C.Rst[i]; j < C.Rst[i+1]-1; ++j)
            assert(C.Col[j] < C.Col[j+1]);
    return C;
}


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
void _mxmm_msa_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
#pragma omp parallel
    {
        MSA<T> accum(B.n);
        T zero = (T)0;

#pragma omp for schedule(dynamic, 32)
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
        std::vector<int> changed_states;
        changed_states.reserve(B.n);
        
#pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < A.m; ++i) {
            int m_begin = M.Rst[i];
            int m_end   = M.Rst[i+1];
            int row_nz = 0;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_begin = B.Rst[k];
                int b_end   = B.Rst[k+1];

                for (int j = b_begin; j < b_end; ++j) {
                    int col = B.Col[j];
                    if (accum.state[col] == MSA<T>::UNALLOWED) {
                        accum.state[col] = MSA<T>::ALLOWED;
                        changed_states.push_back(col);
                        ++row_nz;
                    }
                }
            }
            for (int j = m_begin; j < m_end; ++j) {
                if (accum.state[M.Col[j]] == MSA<T>::ALLOWED)
                    --row_nz;
            }
            C.Rst[i+1] = row_nz;
            
            for (int col_idx: changed_states)
                accum.state[col_idx] = MSA<T>::UNALLOWED;
            changed_states.clear();
        }
#pragma omp single
    {
        C.Rst[0] = 0;
        for (int i = 1; i < A.m; ++i)
            C.Rst[i+1] += C.Rst[i];
        if (C.Rst[A.m] > C.nz)
            C.resizeVals(C.Rst[A.m]);
        C.nz = C.Rst[A.m];
    }

    constexpr T zero = T(0);
    for (size_t i = 0; i < accum.len; ++i)
        accum.state[i] = MSA<T>::ALLOWED;

#pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < A.m; ++i) {
            int m_begin = M.Rst[i];
            int m_end   = M.Rst[i+1];

            for (size_t j = m_begin; j < m_end; ++j)
                accum.state[M.Col[j]] = MSA<T>::UNALLOWED;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_begin = B.Rst[k];
                int b_end   = B.Rst[k+1];
                T   a_val = A.Val[t];

                for (int j = b_begin; j < b_end; ++j) {
                    int col = B.Col[j];
                    if (accum.state[col] == MSA<T>::ALLOWED) {
                        accum.state[col] = MSA<T>::SET;
                        changed_states.push_back(col);
                        accum.value[col] = a_val * B.Val[j];
                    }
                    else if (accum.state[col] == MSA<T>::SET)
                        accum.value[col] += a_val * B.Val[j];
                }
            }
            for (size_t j = m_begin; j < m_end; ++j)
                accum.state[M.Col[j]] = MSA<T>::ALLOWED;
            
            int c_pos = C.Rst[i];
            sort(changed_states.begin(), changed_states.end());
            for (int col_idx : changed_states) {
                C.Col[c_pos] = col_idx;
                C.Val[c_pos++] = accum.value[col_idx];
                accum.value[col_idx] = zero;
                accum.state[col_idx] = MSA<T>::ALLOWED;
            }
            changed_states.clear();
        }
    }
}

template<typename T, typename U>
void _mxmm_msa_cmask_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    MSA<T> accum(B.n);
    std::vector<int> changed_states;
    changed_states.reserve(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];
        int row_nz = 0;

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_min = B.Rst[k];
            int b_max = B.Rst[k+1];

            for (int j = b_min; j < b_max; ++j) {
                if (accum.state[B.Col[j]] == MSA<T>::UNALLOWED) {
                    accum.state[B.Col[j]] = MSA<T>::ALLOWED;
                    changed_states.push_back(B.Col[j]);
                    ++row_nz;
                }
            }
        }
        for (int j = m_min; j < m_max; ++j) {
            if (accum.state[M.Col[j]] == MSA<T>::ALLOWED)
                --row_nz;
        }
        C.Rst[i+1] = row_nz;
        
        for (int col_idx: changed_states)
            accum.state[col_idx] = MSA<T>::UNALLOWED;
        changed_states.clear();
    }
    C.Rst[0] = 0;
    for (int i = 1; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    if (C.Rst[A.m] > C.nz)
        C.resizeVals(C.Rst[A.m]);
    C.nz = C.Rst[A.m];

    constexpr T zero = T(0);
    for (size_t i = 0; i < accum.len; ++i)
        accum.state[i] = MSA<T>::ALLOWED;

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        for (size_t j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                if (accum.state[B.Col[j]] == MSA<T>::ALLOWED) {
                    accum.state[B.Col[j]] = MSA<T>::SET;
                    changed_states.push_back(B.Col[j]);
                    accum.value[B.Col[j]] = a_val * B.Val[j];
                }
                else if (accum.state[B.Col[j]] == MSA<T>::SET)
                    accum.value[B.Col[j]] += a_val * B.Val[j];
            }
        }
        
        int c_pos = C.Rst[i];
        sort(changed_states.begin(), changed_states.end());
        for (int col_idx : changed_states) {
            C.Col[c_pos] = col_idx;
            C.Val[c_pos++] = accum.value[col_idx];
            accum.value[col_idx] = zero;
            accum.state[col_idx] = MSA<T>::ALLOWED;
        }
        changed_states.clear();
        for (size_t j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::ALLOWED;
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

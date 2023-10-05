#pragma once
#include "matrix.h"
#include "matrix_la.h"

std::vector<float> betweenness_centrality(bool isParallel, const spMtx<int> &A, size_t blockSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
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

    std::chrono::high_resolution_clock::time_point 
        eWiseAdd_begin,
        spMul_begin,
        eWiseMult_begin,
        dMul_begin,
        eWiseMultAdd_begin,
        eWiseAdd_end,
        spMul_end,
        eWiseMult_end,
        dMul_end,
        eWiseMultAdd_end;
    long long eWiseAdd_time = 0,
              spMul_time = 0,
              eWiseMult_time = 0,
              dMul_time = 0,
              eWiseMultAdd_time = 0;

    for (size_t i = 0; i < A.n; i += blockSize) {
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

            eWiseAdd_begin = std::chrono::high_resolution_clock::now();
            Numsp = add_nointersect(Numsp, Front);
            eWiseAdd_end = std::chrono::high_resolution_clock::now();
            eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();

            spMul_begin = std::chrono::high_resolution_clock::now();
            mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
            spMul_end = std::chrono::high_resolution_clock::now();
            spMul_time += (spMul_end - spMul_begin).count();

            Front = Fronttmp;
            ++d;
        } while (Front.nz != 0);


        // Обратный проход
        Numspd = Numsp; // преобразование разреженной матрицы в плотную
        Nspinv.resize(m, n);
        Bcu.resize(m, n);
        W.resize(m, n);
        for (size_t i = 0; i < mxn; ++i)
            Bcu.Val[i] = 1.0f;
        for (size_t i = 0; i < mxn; ++i)
            Nspinv.Val[i] = 1.0f / Numspd.Val[i];

        for (size_t k = d-1; k > 0; --k) {
            eWiseMult_begin = std::chrono::high_resolution_clock::now();
            eWiseMult(Nspinv, Bcu, Sigmas[k], W);
            eWiseMult_end = std::chrono::high_resolution_clock::now();
            eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();

            dMul_begin = std::chrono::high_resolution_clock::now();
            mxmm_spd(Af, W, Sigmas[k-1], W);
            dMul_end = std::chrono::high_resolution_clock::now();
            dMul_time += (dMul_end - dMul_begin).count();

            eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
            fuseEWiseMultAdd(W, Numspd, Bcu);
            eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
            eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
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

        std::cerr << "Done " << (i+n)*100ull/A.n << "%\n";
    }

    std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
    std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
    std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";

    return bcv;
}

std::vector<float> betweenness_centrality_batch(bool isParallel, const spMtx<int> &A, size_t batchSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
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

    std::chrono::high_resolution_clock::time_point 
        eWiseAdd_begin,
        spMul_begin,
        eWiseMult_begin,
        dMul_begin,
        eWiseMultAdd_begin,
        eWiseAdd_end,
        spMul_end,
        eWiseMult_end,
        dMul_end,
        eWiseMultAdd_end;
    long long eWiseAdd_time = 0,
              spMul_time = 0,
              eWiseMult_time = 0,
              dMul_time = 0,
              eWiseMultAdd_time = 0;

    size_t n = std::min(A.n, batchSize); // количество столбцов
    size_t mxn = (size_t)m * n;

    Numsp.resizeVals(n);
    Numsp.n = n;
    for (size_t j = 0; j < n; ++j) {
        Numsp.Rst[j] = j;
        Numsp.Col[j] = j;
        Numsp.Val[j] = 1;
    }
    for (size_t j = n; j <= m; ++j)
        Numsp.Rst[j] = n;
    Front = transpose(A.extractRows(0, n));

    // Прямой проход (поиск в ширину)
    size_t d = 0;
    do {
        Sigmas[d] = Front;

        eWiseAdd_begin = std::chrono::high_resolution_clock::now();
        Numsp = add_nointersect(Numsp, Front);
        eWiseAdd_end = std::chrono::high_resolution_clock::now();
        eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();

        spMul_begin = std::chrono::high_resolution_clock::now();
        mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
        spMul_end = std::chrono::high_resolution_clock::now();
        spMul_time += (spMul_end - spMul_begin).count();

        Front = Fronttmp;
        ++d;
    } while (Front.nz != 0);


    // Обратный проход
    Numspd = Numsp; // преобразование разреженной матрицы в плотную
    Nspinv.resize(m, n);
    Bcu.resize(m, n);
    W.resize(m, n);
    for (size_t i = 0; i < mxn; ++i)
        Bcu.Val[i] = 1.0f;
    for (size_t i = 0; i < mxn; ++i)
        Nspinv.Val[i] = 1.0f / Numspd.Val[i];

    for (size_t k = d-1; k > 0; --k) {
        eWiseMult_begin = std::chrono::high_resolution_clock::now();
        eWiseMult(Nspinv, Bcu, Sigmas[k], W);
        eWiseMult_end = std::chrono::high_resolution_clock::now();
        eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();

        dMul_begin = std::chrono::high_resolution_clock::now();
        mxmm_spd(Af, W, Sigmas[k-1], W);
        dMul_end = std::chrono::high_resolution_clock::now();
        dMul_time += (dMul_end - dMul_begin).count();

        eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
        fuseEWiseMultAdd(W, Numspd, Bcu);
        eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
        eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
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

    std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
    std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
    std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";

    return bcv;
}
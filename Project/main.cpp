#include <ctime>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <direct.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include "graphio_cpp.h"
#include "my_sparse.h"
#include "spmv.h"
using namespace std;

sparse_matrix_t create_mkl_spm(const crsGraphT<double> &gr) {
    int status;
    sparse_matrix_t mkl_spm;
    if (status = mkl_sparse_d_create_csr(&mkl_spm, SPARSE_INDEX_BASE_ZERO, gr.v, gr.v, gr.Xdj, gr.Xdj + 1, gr.Adj, gr.Wgt))
        exit(status);

    return mkl_spm;
}

crsGraphT<double> mkl_mspmm(const crsGraphT<double> &A, const crsGraphT<double> &B, const crsGraphT<double> &M) {
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

    crsGraphT<double> C;
    sparse_index_base_t indexing;
    mkl_sparse_d_export_csr(mkl_product, &indexing, &C.v, &C.v, &C.Xdj, &C.Xdj + 1, &C.Adj, &C.Wgt);
    C.nz = C.Xdj[n];

    int c_last = 0;
    int c_curr = 0;
    for (int i = 0; i < C.v; ++i) {
        for (int j = M.Xdj[i]; j < M.Xdj[i+1]; ++j) {
            while (c_curr < C.Xdj[i] && C.Adj[c_curr] < M.Adj[j])
                ++c_curr;
            if (c_curr < C.Xdj[i+1] && C.Adj[c_curr] == M.Adj[j]) {
                C.Adj[c_last] = C.Adj[c_curr];
                C.Wgt[c_last++] = C.Wgt[c_curr++];
            }
        }
        C.Xdj[i+1] = c_last;
    }


    //mkl_sparse_destroy(mkl_sp_a);
    //mkl_sparse_destroy(mkl_sp_b);
    //mkl_sparse_destroy(mkl_product);
    return C;
}

int* triangle_counting(const crsGraphT<int> &gr) {
    int *nums_of_tr = new int [gr.v];
    crsGraphT<int> square; // квадрат матрицы смежности
    crsGraphT<int> mask;
    full_mask(mask, gr.v);

    mspgemm_mca(gr, gr, mask, square);
    
    // подсчёт для каждой вершины числа треугольников, в которые она входит
    for (int i = 0; i < gr.v; ++i) {
        int num_of_tr = 0;
        int gr_curr_pos = gr.Xdj[i];
        for (int j = square.Xdj[i]; j < square.Xdj[i+1]; ++j) {
            while (gr_curr_pos < gr.Xdj[i+1] && gr.Adj[gr_curr_pos] < square.Adj[j])
                ++gr_curr_pos;
            if (gr_curr_pos < gr.Xdj[i+1] && gr.Adj[gr_curr_pos] == square.Adj[j])
                num_of_tr += gr.Wgt[gr_curr_pos] * square.Wgt[j];
        }
        nums_of_tr[i] = num_of_tr;
    }
    for (int i = 0; i < gr.v; ++i)
        nums_of_tr[i] >>= 1;
    
    return nums_of_tr;
}






string input_path(const char *filename) {
    string logfile("../graphs/");
    logfile += filename;
    return logfile;
}

string output_path() {
    std::time_t current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    const char *current_date_time = ctime(&current_time);
    string logfile("../logs/");
    logfile += current_date_time;
    logfile.pop_back();
    replace(logfile.begin(), logfile.end(), ' ', '_');
    replace(logfile.begin(), logfile.end(), ':', '-');
    logfile += ".txt";
    return logfile;
}

int main(int argc, const char* argv[]) {
    //if (argc < 2)
    //    return -2;
    crsGraphT<int> gr("../graphs/cage3unweighted.mtx");
    cout << "Finished reading\n";
    //if (freopen(output_path().c_str(), "w", stdout) == NULL)
    //    return -1;


    int *nums_of_tr = triangle_counting(gr);
    for (int i = 0; i < gr.v; ++i)
        cerr << nums_of_tr[i] << ' ';
    cerr << '\n';
    delete[] nums_of_tr;
}

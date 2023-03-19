#define PARALLEL

#include "spMatrix.h"
#include "my_sparse.h"
#include <omp.h>
#include <cmath>
#include <ctime>
#include <functional>
using namespace std;

template <typename T>
using mxmOp = spMatrix<T>(*)(const spMatrix<T>&, const spMatrix<T>&, const spMatrix<T>&);

int* triangle_counting_vertex(const spMatrix<int> &A, mxmOp<int> matrixMult) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.v];
    int num_of_tr;
    spMatrix<int> SQ; // A^2 (A is adjacency matrix)

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    SQ = matrixMult(A, A, A);
    // for each vertex we count the number of triangles it belongs to
    for (int i = 0; i < A.v; ++i) {
        num_of_tr = 0;
        for (int j = SQ.Rst[i]; j < SQ.Rst[i+1]; ++j)
            num_of_tr += SQ.Val[j];
        nums_of_tr[i] = num_of_tr >>= 1;
    }
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Vertices: " << A.v << '\n';
    cout << "Edges: " << A.nz << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << '\n';

    return nums_of_tr;
}

// mspgemm_naive
// mspgemm_naive_parallel
// mspgemm_mca
// mspgemm_mca_parallel
// mspgemm_heap
// mspgemm_heap_parallel

int64_t triangle_counting_masked_lu(const spMatrix<int> &A, mxmOp<int> matrixMult) {
    int64_t num_of_tr = 0;
    spMatrix<int> L = extract_lower_triangle(A);
    spMatrix<int> U = transpose(L);
    spMatrix<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    C = matrixMult(L, U, A);

    // Count the total number of triangles
    for (int j = 0; j < C.Rst[C.v]; ++j)
            num_of_tr += C.Val[j];
    num_of_tr >>= 1;
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "TRIANGLE COUNTING (LU)\n";
    cout << "Vertices: " << A.v << '\n';
    cout << "Edges: " << A.nz << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Number of triangles: " << num_of_tr << '\n';

    return num_of_tr;
}

int64_t triangle_counting_masked_sandia(const spMatrix<int> &A, mxmOp<int> matrixMult) {
    int64_t num_of_tr = 0;
    spMatrix<int> L = extract_lower_triangle(A);
    spMatrix<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    C = matrixMult(L, L, L);

    // Count the total number of triangles
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.v]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "TRIANGLE COUNTING (SANDIA)\n";
    cout << "Vertices: " << A.v << '\n';
    cout << "Edges: " << A.nz << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Number of triangles: " << num_of_tr << '\n';

    return num_of_tr;
}

/* K-TRUSS */
spMatrix<int> k_truss(const spMatrix<int> &A, int k, mxmOp<int> matrixMult) {
    spMatrix<int> C = A;  // a copy of adjacency matrix
    spMatrix<int> Tmp;
    int n = A.v;
    int *tmp_Xdj = new int[n+1];
    tmp_Xdj[0] = 0;

    auto start = chrono::steady_clock::now();

    for (int t = 0; t < n; ++t) {
        // Tmp<C> = C*C
        Tmp = matrixMult(C, C, C);

        // remove all edges included in less than (k-2) triangles
        // and replace values of remaining entries with 1
        int new_curr_pos = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = Tmp.Rst[i]; j < Tmp.Rst[i+1]; ++j) {
                if (Tmp.Val[j] >= k-2) {
                    Tmp.Col[new_curr_pos]   = Tmp.Col[j];
                    Tmp.Val[new_curr_pos++] = 1;
                }
            }
            tmp_Xdj[i+1] = new_curr_pos;
        }
        memcpy(Tmp.Rst, tmp_Xdj, (n+1)*sizeof(int));
        Tmp.nz = Tmp.Rst[n];

        // check if the number of edges has changed
        if (Tmp.nz == C.nz)
            break;

        // Assign 'Tmp' to 'C'
        C = std::move(Tmp);
    }

    auto finish = chrono::steady_clock::now();
    cout << "K-TRUSS " << k << '\n';
    cout << "Vertices: " << A.v << '\n';
    cout << "Edges: " << A.nz << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";

    if (C.nz < A.nz) {
        int *new_Adj = new int[C.nz];
        int *new_Wgt = new int[C.nz];
        std::memcpy(new_Adj, C.Col, C.nz * sizeof(int));
        std::memcpy(new_Wgt, C.Val, C.nz * sizeof(int));
        delete[] C.Col;
        delete[] C.Val;
        C.Col = new_Adj;
        C.Val = new_Wgt;
    }

    delete[] tmp_Xdj;
    return C;
}

string output_path(const string &graph_name, bool from_vs) {
    std::time_t current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    const char *current_date_time = ctime(&current_time);
    string logfile(from_vs ? "../logs/" : "../../logs/");
    logfile += current_date_time;
    logfile.pop_back();
    replace(logfile.begin(), logfile.end(), ' ', '_');
    replace(logfile.begin(), logfile.end(), ':', '-');
    logfile += "_" + graph_name;
    logfile += ".txt";
    return logfile;
}

int main(int argc, const char* argv[]) {
    // cout << input_path(argv[1]).c_str() << '\n';
    spMatrix<double> gr((argc > 1) ? argv[1] : "../graphs/EAT_SR.mtx", false);
    cerr << "finished reading\n";
    // freopen(output_path((argc > 1) ? argv[1] : "EAT_SR", argc < 2).c_str(), "w", stdout);
    
    /*
    spMatrix<int> AdjMatrix = build_symm_from_lower(extract_lower_triangle(build_adjacency_matrix(gr)));
    spMatrix<int> L = extract_lower_triangle(AdjMatrix);
    spMatrix<int> X, Y;
    
    mspgemm_mca_parallel(L, L, L, X);
    mspgemm_heap_parallel(L, L, L, Y);

    for (int i = 0; i < X.v; ++i) {
        for (int j = X.Rst[i]; j < X.Rst[i+1]; ++j) {
            if (X.Val[j] != Y.Val[j])
                cerr << '(' << i << ", " << X.Col[j] << "): (" << X.Val[j] << ", " << Y.Val[j] << ")\n";
        }
    }
    */

#ifdef PARALLEL
    cout << "PARALLEL NAIVE " << omp_get_max_threads() << "\n\n";
    cerr << "PARALLEL NAIVE " << omp_get_max_threads() << "\n\n";
#else
    cout << "SEQUENTIAL NAIVE\n";
    cerr << "SEQUENTIAL NAIVE\n";
#endif

    /* DEBUG PART */
    /*
    spMatrix<int> Col = generate_adjacency_matrix(7,3,6);
    spMatrix<int> L = extract_lower_triangle(Col);
    spMatrix<int> C;
    Col.print_dense();
    mspgemm_heap(L, L, L, C);
    C.print_dense();
    */
    spMatrix<int> Col = build_symm_from_lower(extract_lower_triangle(build_adjacency_matrix(gr)));

    triangle_counting_masked_sandia(Col, mxmm_mca_par);
    cout << "\n\n";
    cerr << "triangle sandia done\n";
    
    triangle_counting_masked_lu(Col, mxmm_mca_par);
    cout << "\n\n";
    cerr << "triangle lu done\n";
    /*
    k_truss(AdjMatrix, 3);
    cout << "\n\n";
    cerr << "k-truss " << 3 << " done\n";

    k_truss(AdjMatrix, 5);
    cout << "\n\n";
    cerr << "k-truss " << 5 << " done\n";
    */
    
    cerr << "finished program\n";
    return 0;
}

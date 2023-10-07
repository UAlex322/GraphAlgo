#include "matrix.h"
#include "betweenness_centrality.h"
#include <omp.h>
#include <cmath>
#include <ctime>
#include <sstream>
#include <functional>
#include "bin_reader.h"
using namespace std;

template <typename T>
using mxmOp = void(*)(bool, const spMtx<T>&, const spMtx<T>&, const spMtx<T>&, spMtx<T>&);

int* triangle_counting_vertex(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.m];
    int num_of_tr;
    spMtx<int> SQ; // A^2 (A is adjacency matrix)

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, A, A, A, SQ);
    // for each vertex we count the number of triangles it belongs to
    for (size_t i = 0; i < A.m; ++i) {
        num_of_tr = 0;
        for (int j = SQ.Rst[i]; j < SQ.Rst[i+1]; ++j)
            num_of_tr += SQ.Val[j];
        nums_of_tr[i] = num_of_tr >>= 1;
    }
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << '\n';

    return nums_of_tr;
}


int64_t triangle_counting_masked_lu(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    spMtx<int> L = extract_lower_triangle(A);
    spMtx<int> U = transpose(L);
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, U, A, C);

    // Count the total number of triangles
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    num_of_tr >>= 1;
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}


int64_t triangle_counting_masked_sandia(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    spMtx<int> L = extract_lower_triangle(A);
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, L, L, C);

    // Count the total number of triangles
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}


/* K-TRUSS */
spMtx<int> k_truss(const spMtx<int> &A, int k, mxmOp<int> matrixMult, bool isParallel) {
    spMtx<int> C = A;  // a copy of adjacency matrix
    spMtx<int> Tmp;
    int n = A.m;
    int totalIterationNum = 0;
    int *tmp_Xdj = new int[n+1];
    tmp_Xdj[0] = 0;

    auto start = chrono::steady_clock::now();

    for (int t = 0; t < n; ++t) {
        // Tmp<C> = C*C
        matrixMult(isParallel, C, C, C, Tmp);

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
        if (Tmp.nz == C.nz) {
            totalIterationNum = ++t;
            break;
        }

        // Assign 'Tmp' to 'C'
        std::swap(C, Tmp);
    }

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Iterations: " << totalIterationNum << '\n';

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


GraphInfo get_graph_info(int argc, const char *argv[]) {
    std::time_t current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    string current_date_time(ctime(&current_time));
    string graphName;
    string graphPath(argv[1]);
    string logfile(argv[2]);
    string action(argv[3]);
    string format;

    int slashPos = graphPath.size() - 1;
    int dotPos   = graphPath.size() - 1;

    while (graphPath[slashPos] != '/')
        --slashPos;
    while (graphPath[dotPos] != '.')
        --dotPos;

    graphName = graphPath.substr(slashPos + 1, dotPos - slashPos - 1);
    format = graphPath.substr(dotPos + 1, graphPath.size() - dotPos);
    graphPath = graphPath.substr(0, slashPos + 1);

    if (action == "launch") {
        current_date_time.pop_back();
        replace(current_date_time.begin(), current_date_time.end(), ' ', '_');
        replace(current_date_time.begin(), current_date_time.end(), ':', '-');

        stringstream ss;
        ss << logfile << '/' << current_date_time << '_' << graphName;
        for (int i = 4; i < argc; ++i)
            ss << '_' << argv[i];
        ss << '_' << to_string(omp_get_max_threads()) << ".txt";
        logfile = ss.str();
    }

    GraphInfo info {graphName, graphPath, logfile, format};

    return info;
}

int launch_test(const spMtx<int> &gr, const GraphInfo &info, int argc, const char *argv[]) {
    string benchmarkAlgorithm(argv[4]),
           parOrSeq(argv[5]),
           batchSizeStr;
    bool isParallel = (parOrSeq == "par");
    size_t batch_size;
    mxmOp<int> mxm_algorithm;
    spMtx<int> MxmResult, TestMtx = gr;
    chrono::high_resolution_clock::time_point start, finish, default_time;
    stringstream alg_ss;

    if (benchmarkAlgorithm == "bc") {
        if (argc < 7 || (batch_size = atoll(argv[6])) == 0) {
            cerr << "incorrect input, 6-th argument: batch has to be positive integer\n";
            return -6;
        }
    }
    else {
        string multiplicationAlgorithm(argv[6]);
        if (multiplicationAlgorithm == "naive")
            mxm_algorithm = mxmm_naive<int>;
        if (multiplicationAlgorithm == "msa")
            mxm_algorithm = mxmm_msa<int>;
        else if (multiplicationAlgorithm == "mca")
            mxm_algorithm = mxmm_mca<int>;
        else if (multiplicationAlgorithm == "heap")
            mxm_algorithm = mxmm_heap<int>;
        else {
            cerr << "incorrect input, 6-th argument: has to be 'naive', 'msa', 'mca' or 'heap')\n";
            return -7;
        }
    }

    if (benchmarkAlgorithm != "bc")
        TestMtx = build_symm_from_lower(extract_lower_triangle(TestMtx));

    if (parOrSeq == "par") {
        alg_ss << "Parallel,   " << omp_get_max_threads() << " threads\n";
        // cerr << "PARALLEL " << omp_get_max_threads() << " THREADS\n";
    } else if (parOrSeq == "seq") {
        alg_ss << "Sequential\n";
        // cerr << "SEQUENTIAL \n\n";
    } else {
        cerr << "incorrect input, 5-th argument: has to be 'par' or 'seq'\n";
        return -5;
    }

    if (benchmarkAlgorithm == "mxm") {
        start = chrono::high_resolution_clock::now();
        mxm_algorithm(isParallel, TestMtx, TestMtx, TestMtx, MxmResult);
        finish = chrono::high_resolution_clock::now();
        alg_ss << "Algorithm:  matrix square\n";
    }
    else if (benchmarkAlgorithm == "k-truss") {
        if (argc < 8 || atoi(argv[7]) < 3) {
            cerr << "incorrect input, 7-th argument: has to be positive integer bigger than 2\n";
        }
        alg_ss << "Algorithm:  k-truss, k = " << argv[7] << '\n';
        k_truss(TestMtx, stoi(argv[7]), mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "triangle") {
        alg_ss << "Algorithm:  triangle counting\n";
        triangle_counting_masked_sandia(TestMtx, mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "bc") {
        vector<float> bcVector;
        start = chrono::high_resolution_clock::now();
        // size_t cache_fit_size = 1747626; // to size of float matrix to fit into 20 Mb cache 
        // size_t batch_size = (cache_fit_size/Adj.m > 0) ? cache_fit_size/Adj.m : 3;
        bcVector = betweenness_centrality_batch(isParallel, TestMtx, batch_size);
        // vector<float> bcVector = betweenness_centrality(isParallel, TestMtx, 5);
        // for (int i = 0; i < bcVector.size(); ++i)
        //     cout << i << " : " << bcVector[i] << '\n';
        // cout << '\n';
        finish = chrono::high_resolution_clock::now();
        alg_ss << "Algorithm:  betweenness centrality\n" << "Batch size: " << batch_size << '\n';
    }
    else {
        cerr << "incorrect input, 4-th argument: has to be 'triangle', 'k-truss', 'mxm' or 'bc')\n";
        return -4;
    }
    long long time = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    if (start != default_time)
        cout << "Time:       " << time << " ms\n";
    cout << "Graph name: " << info.graphName << '\n';
    cout << "Vertices:   " << TestMtx.m << '\n';
    cout << "Edges:      " << TestMtx.nz << '\n';
    cout << alg_ss.str() << '\n';
    return 0;
}

string get_graph_val_type(const char *filename, const GraphInfo &info) {
    ifstream istr(filename, (info.format == "bin") ? std::ios::in | std::ios::binary
                                                   : std::ios::in);
    string stype;
    if (info.format == "bin") {
        char type;
        istr >> type >> type >> type;
        if (type == 'R')
            stype = "real";
        else if (type == 'I' || type == 'P')
            stype = "integer";
    } else if (info.format == "mtx") {
        string type;
        istr >> type >> type >> type >> type;
        if (type == "complex")
            throw "Can't use complex numbers!";
        if (type == "real")
            stype = "real";
        else
            stype = "integer";
    } else if (info.format == "graph")
        stype = "integer";
    else if (info.format == "rmat")
        stype = "integer";
    else {
        istr.close();
        throw "Unknown format";
    }

    istr.close();
    return stype;
}

// argv[1] - ����
// argv[2] - ����� ��� ������ �����
// argv[3] - ����������� �������� (tobinary / launch)
// ��� 'launch':
//   argv[4] - ����������� �������� (triangle / k-truss / mxm / bc)
//   argv[5] - ��������������� ��� ����������� (seq / par)
//   ��� 'bc':
//     argv[6] - batch size (����� ������, ��� ������� ����������� ��������)
//   ��� 'triangle' / 'k-truss' / 'mxm':
//     argv[6] - ������������ �������� ��������� (naive / msa / mca / heap)
//     ��� k-truss:
//       argv[7] - �������� 'k' � k-truss

template <typename ValType>
int read_graph_and_launch_test(const GraphInfo &info, int argc, const char *argv[]) {
    string action(argv[3]);
    spMtx<ValType> gr(argv[1], info.format.c_str());
    cerr << "finished reading\n";
    if (action == "tobinary") {
        gr.write_crs_to_bin((info.graphPath + info.graphName + ".bin").c_str());
        cerr << "finished writing to BIN\n";
        return 0;
    } else if (action == "launch") {
       return launch_test(build_adjacency_matrix(gr), info, argc, argv);
    } else {
        cerr << "incorrect input (3-nd argument has to be 'tobinary' or 'launch')\n";
        return -3;
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        cerr << "not enough arguments (have to be at least 3)\n";
        return -1;
    }

    GraphInfo info = get_graph_info(argc, argv);
    string grType = get_graph_val_type(argv[1], info);

    if (grType == "integer")
        return read_graph_and_launch_test<int>(info, argc, argv);
    else if (grType == "real") {
        return read_graph_and_launch_test<double>(info, argc, argv);
    } else {
        cerr << "unknown value type\n";
        return -2;
    }

    return 0;
}

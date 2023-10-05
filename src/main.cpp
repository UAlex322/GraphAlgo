#include "matrix.h"
#include "betweenness_centrality.h"
#include <omp.h>
#include <cmath>
#include <ctime>
#include <sstream>
#include <functional>
#include "bin_reader.h"
using namespace std;

string globalGraphName;

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
    cout << "Graph: " << globalGraphName << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << '\n';

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
    cout << "TRIANGLE COUNTING (LU)\n";
    cout << "Graph: " << globalGraphName << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Number of triangles: " << num_of_tr << '\n';

    return num_of_tr;
}


int64_t triangle_counting_masked_sandia(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    spMtx<int> L = extract_lower_triangle(A);
    spMtx<int> C;

    cout << A.nz << '\n';
    cout << L.nz << '\n';

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, L, L, C);

    // Count the total number of triangles
#pragma omp parallel for reduction(+:num_of_tr)
    for (int j = 0; j < C.Rst[C.m]; ++j)
        num_of_tr += C.Val[j];
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "TRIANGLE COUNTING (SANDIA)\n";
    cout << "Graph: " << globalGraphName << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Number of triangles: " << num_of_tr << '\n';

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
    cout << "K-TRUSS " << k << '\n';
    cout << "Graph: " << globalGraphName << '\n';
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Number of iterations: " << totalIterationNum << '\n';

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
    globalGraphName = graphName;

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

template<typename T>
int launch_test(const spMtx<T> &gr, const char *argv[]) {
    string benchmarkAlgorithm(argv[4]),
           parOrSeq(argv[5]),
           batchSizeStr;
    bool isParallel = (parOrSeq == "par");
    mxmOp<int> mxm_algorithm;
    size_t batch_size;

    if (benchmarkAlgorithm == "bc") {
        batch_size = atoll(argv[6]);
        if (batch_size == 0) {
            cerr << "incorrect input, 6-th argument: has to be positive integer\n";
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
            return -3;
        }
    }

    spMtx<int> MxmResult, Symm;
    if (benchmarkAlgorithm != "bc")
        Symm = build_symm_from_lower(extract_lower_triangle(build_adjacency_matrix(gr)));

    if (parOrSeq == "par") {
        cout << "PARALLEL " << omp_get_max_threads() << " THREADS\n";
        // cerr << "PARALLEL " << omp_get_max_threads() << " THREADS\n";
    } else if (parOrSeq == "seq") {
        cout << "SEQUENTIAL\n";
        // cerr << "SEQUENTIAL \n\n";
    } else {
        cerr << "incorrect input, 5-th argument: has to be 'par' or 'seq')\n";
        return -2;
    }

    if (benchmarkAlgorithm == "mxm") {
        auto start = chrono::steady_clock::now();
        mxm_algorithm(isParallel, Symm, Symm, Symm, MxmResult);
        auto finish = chrono::steady_clock::now();
        cout << "MATRIX SQUARE\n" << "Vertices: " << Symm.m << "\nEdges: " << Symm.nz
            << "\nTime: " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n\n";
    }
    else if (benchmarkAlgorithm == "k-truss") {
        k_truss(Symm, stoi(argv[7]), mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "triangle")
        triangle_counting_masked_sandia(Symm, mxm_algorithm, isParallel);
    else if (benchmarkAlgorithm == "bc") {
        auto Adj = build_adjacency_matrix(gr);
        auto start = chrono::steady_clock::now();
        // size_t cache_fit_size = 1747626; // to size of float matrix to fit into 20 Mb cache 
        // size_t batch_size = (cache_fit_size/Adj.m > 0) ? cache_fit_size/Adj.m : 3;
        vector<float> bcVector = betweenness_centrality_batch(isParallel, Adj, batch_size);
        // for (int i = 0; i < bcVector.size(); ++i)
        //     cout << i << " : " << bcVector[i] << '\n';
        // cout << '\n';
        auto finish = chrono::steady_clock::now();
        cout << "BETWEENNESS CENTRALITY" << "\nBatch size: " << batch_size << "\nTime: " << 
                chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n\n";
    }
    else {
        cerr << "incorrect input, 4-th argument: has to be 'triangle', 'k-truss', 'mxm' or 'bc')\n";
        return -4;
    }
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

// argv[1] - граф
// argv[2] - папка для вывода логов
// argv[3] - выполняемая операция (tobinary / launch)
// для 'launch':
//   argv[4] - выполняемый алгоритм (triangle / k-truss / mxm / bc)
//   argv[5] - последовательно или параллельно (seq / par)
//   для 'bc':
//     argv[6] - batch size (число вершин, для которых запускается алгоритм)
//   для 'triangle' / 'k-truss' / 'mxm':
//     argv[6] - используемый алгоритм умножения (naive / msa / mca / heap)
//     для k-truss:
//       argv[7] - параметр 'k' в k-truss
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        cerr << "not enough arguments (have to be at least 3)\n";
        return -1;
    }

    string action(argv[3]);
    
    GraphInfo info = get_graph_info(argc, argv);
    string grType = get_graph_val_type(argv[1], info);

    if (grType == "integer") {
        spMtx<int> gr(argv[1], info.format.c_str());
        cerr << "finished reading\n";
        if (action == "tobinary") {
            gr.write_crs_to_bin((info.graphPath + info.graphName + ".bin").c_str());
            cerr << "finished writing to BIN\n";
        } else if (action == "launch") {
           return launch_test(gr, argv);
        } else {
            cerr << "incorrect input (2-nd argument has to be 'tobinary' or 'launch')\n";
            return -5;
        }
    } else if (grType == "real") {
        spMtx<double> gr(argv[1], info.format.c_str());
        cerr << "finished reading\n";
        if (action == "tobinary") {
            gr.write_crs_to_bin((info.graphPath + info.graphName + ".bin").c_str());
            cerr << "finished writing to BIN\n";
        } else if (action == "launch") {
            return launch_test(gr, argv);
        } else {
            cerr << "incorrect input (2-nd argument has to be 'tobinary' or 'launch')\n";
            return -5;
        }
    } else {
        cerr << "unknown value type\n";
        return -5;
    }

    return 0;
}

#ifndef MY_SPARSE_H 
#define MY_SPARSE_H

#include "graphio_cpp.h"
#include <cstring>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <random>
#include <set>
#include <queue>

crsGraphT<int> generate_mask(const int n, const int min_deg, const int max_deg) {
	std::set<int> positions;
	std::uniform_int_distribution<int> deg_distr(min_deg, max_deg);
	std::uniform_int_distribution<int> col_distr(0, n - 1);
	std::mt19937 generator{std::random_device{}()};
	crsGraphT<int> mask;
	mask.Xdj = new int[n+1];
	mask.v = n;

	mask.Xdj[0] = 0;
	for (int i = 0; i < n; ++i)
		mask.Xdj[i+1] = mask.Xdj[i] + deg_distr(generator);
	mask.nz = mask.Xdj[n];
	mask.Adj = new int[mask.nz];

	int j = 0;
	for (int i = 0; i < mask.v; ++i) {
		size_t deg = mask.Xdj[i+1] - mask.Xdj[i];
		while (positions.size() < deg)
			positions.insert(col_distr(generator));
		for (const int column : positions)
			mask.Adj[j++] = column;
		positions.clear();
	}

	return mask;
}

template<typename T>
void full_mask(crsGraphT<T> &mask, const int n) {
	mask.v = n;
	mask.nz = n*n;
	mask.Xdj = new int[n+1];
	mask.Adj = new int[(int64_t)n*n];

	for (int i = 0; i <= n; ++i)
		mask.Xdj[i] = i*n;
	int64_t curr_pos = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			mask.Adj[curr_pos++] = j;
	}
}

/*MSpGEMM � �������������� ������� (�����������) ������������*/
template <typename T>
struct MCA {
	static enum {ALLOWED = 0, SET} mca_states;
	char   *states;
	T      *values;
	size_t  len;

	MCA(size_t n) {
		values = new T[n];
		states = new char[n];
		   len = n;

		std::memset(values, 0, len * sizeof(T));
		std::memset(states, 0, len * sizeof(char));
	}

	~MCA() {
		delete[] values;
		delete[] states;
	}

	void clear() {
		std::memset(values, 0, len * sizeof(T));
		std::memset(states, 0, len * sizeof(char));
	}
};

template<typename T>
void mspgemm_mca(const crsGraphT<T> &A, const crsGraphT<T> &B, const crsGraphT<T> &M, crsGraphT<T> &C) {
	// ������������� C
	C.nz     = M.nz;
	C.v      = M.v;
	C.Adj    = new int[M.nz];
	C.Xdj    = new int[M.v + 1];
	C.Xdj[0] = 0;
	C.Wgt    = new T[C.nz];
	std::memset(C.Wgt, 0, C.nz * sizeof(double));

	// ���������� ������������ ����� ������ �����,
	// ����� �������� ��������������� ���������� ������
	
	int max_len = 0;
	for (int i = 0; i < A.v; ++i)
		if (M.Xdj[i+1] - M.Xdj[i] > max_len)
			max_len = M.Xdj[i+1] - M.Xdj[i];

	MCA<T> accum(max_len);

	for (int i = 0; i < A.v; ++i) {
		int m_row_len  = M.Xdj[i+1] - M.Xdj[i];
		int m_row_curr;
		// ������� i-� ������ ������� C
		for (int t = A.Xdj[i]; t < A.Xdj[i+1]; ++t) {
			int k = A.Adj[t];
			int b_row_curr = B.Xdj[k];
			m_row_curr = M.Xdj[i];
			// ����������� ������ � ��������� ������ �������� � ����� ���������
			for (int j = 0; j < m_row_len; ++j, ++m_row_curr) {
				// ���� ��������� �������� � ����� �������
				while (b_row_curr < B.Xdj[k+1] && B.Adj[b_row_curr] < M.Adj[m_row_curr])
					++b_row_curr;
				// ��� ���������� ����������� ��������
				if (b_row_curr < B.Xdj[k+1] && B.Adj[b_row_curr] == M.Adj[m_row_curr]) {
					accum.states[j] = MCA<T>::SET;
					accum.values[j] += A.Wgt[t] * B.Wgt[b_row_curr];
				}
			}
		}
		// ���������� i-� ������ ������� C
		int c_row_curr = C.Xdj[i];
		m_row_curr = M.Xdj[i];
		for (int j = 0; j < m_row_len; ++j, ++m_row_curr) {
			if (accum.states[j] == MCA<T>::SET) {
				C.Adj[c_row_curr] = M.Adj[m_row_curr];
				C.Wgt[c_row_curr++] = accum.values[j];
			}
		}
		C.Xdj[i+1] = c_row_curr;
		// ������� ������������ ��� ��������� ��������
		accum.clear();
	}

	C.nz = C.Xdj[C.v];
	
}

/*MSpGEMM � �������������� ����*/

/*�������� ��� �������� ���������� � ������� �������� ������ ������� B*/
struct spm_iterator {
	int a_pos; // ������� �������� ������ ������� � � �������
	int b_pos; // ������� �������� ������� ������� B � �������
	int b_col; // ������ �������� ������� ������� B � �������

	spm_iterator(int x, int y, int z):
		a_pos(x), b_pos(y), b_col(z) {}
	spm_iterator(const spm_iterator &it):
		a_pos(it.a_pos), b_pos(it.b_pos), b_col(it.b_col) {}
};

bool operator<(const spm_iterator &it1, const spm_iterator it2) {
	return it1.b_col > it2.b_col;
}


template<typename T>
void mspgemm_heap(const crsGraphT<T> &A, const crsGraphT<T> &B, const crsGraphT<T> &M, crsGraphT<T> &C) {
	// ������������� C
	C.nz = M.nz;
	C.v  = M.v;
	C.Adj = new int[C.nz];
	C.Xdj = new int[C.v + 1];
	C.Wgt = new   T[C.nz];
	C.Xdj[0]  = 0;
	std::memset(C.Adj, -1, C.nz * sizeof(int));
	std::memset(C.Wgt,  0, C.nz * sizeof(T));

	int m_curr;     // ������� ������� � ����� �
	int m_max_pos;  // ������� ��� ������� ������ ����� M
	int c_curr = 0; // ������� ������� � �������� ������� C
	for (int i = 0; i < A.v; ++i) {
		std::priority_queue<spm_iterator> heap;

		// ���������� ����
		// k - ������� ������ ������ A.Adj[j] � ������� ������� B
		for (int j = A.Xdj[i]; j < A.Xdj[i+1]; ++j) {
			int k = B.Xdj[A.Adj[j]];
			heap.emplace(j, k, B.Adj[k]);
		}

		m_curr = M.Xdj[i];
		m_max_pos = M.Xdj[i+1];
		while (!heap.empty()) {
			// ������ �������� � ����������� �������� ������� B
			spm_iterator iter = heap.top();
			heap.pop();

			// ���� ������� � � ������ ������������ �������� ������� � B, ���� ������ �� ������� �������
			while (m_curr < m_max_pos && M.Adj[m_curr] < iter.b_col) {
				++m_curr;
				if (C.Adj[c_curr] != -1)
					++c_curr;
			}
			// ���� ����� �� �����, �������
			if (m_curr == m_max_pos)
				break;
			// ��� ���������� �������� � M � B �������� � ���������� ���������
			if (M.Adj[m_curr] == iter.b_col) {
				C.Adj[c_curr] = A.Adj[iter.b_pos];
				C.Wgt[c_curr] += A.Wgt[iter.a_pos] * B.Wgt[iter.b_pos];
			}

			// ��������� ������� ��������� ������� � ����
			int b_max_pos = B.Xdj[A.Adj[iter.a_pos] + 1];
			// ���� � ��������� ������� B ������, ��� ������� ������� M, �����������, ���� �� ������
			++iter.b_pos;
			iter.b_col = B.Adj[iter.b_pos];
			int m_curr_pos = M.Adj[m_curr];
			while (iter.b_pos < b_max_pos && B.Adj[iter.b_pos] < m_curr_pos) {
				++iter.b_pos;
				iter.b_col = B.Adj[iter.b_pos];
			}
			// ��������, �������� �� ����� ������ � B, �� �����������
			if (iter.b_pos < b_max_pos) {
				iter.b_col = B.Adj[iter.b_pos];
				heap.push(iter);
			}
		}
		if (C.Adj[c_curr] != -1)
			++c_curr;
		C.Xdj[i+1] = c_curr;
	}
	C.nz = C.Xdj[A.v];
}



#endif
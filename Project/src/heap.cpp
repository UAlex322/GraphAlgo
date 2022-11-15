#include "heap.h"
#include <cstring>
#include <queue>
using namespace std;

struct spm_iterator {
	int a_pos; // позиция элемента строки матрицы А в массиве
	int b_pos; // позиция элемента столбца матрицы B в массиве
	int b_col; // индекс элемента столбца матрицы B в массиве

	spm_iterator(int x, int y, int z):
		a_pos(x), b_pos(y), b_col(z) {}
	spm_iterator(const spm_iterator &it):
		a_pos(it.a_pos), b_pos(it.b_pos), b_col(it.b_col) {}
};

bool operator<(const spm_iterator &it1, const spm_iterator it2) {
	return it1.b_col > it2.b_col;
}

void mspmm_heap(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C) {
	// инициализация C
	C->nz = M->nz;
	C->V  = M->V;
	C->Adjncy   = (int*)malloc(sizeof(int) * C->nz);
	C->Xadj     = (int*)malloc(sizeof(int) * (C->V + 1));
	C->Eweights = (double*)malloc(sizeof(double) * C->nz);
	C->Xadj[0]  = 0;
	memset(C->Adjncy, -1, C->nz * sizeof(int));
	memset(C->Eweights, 0, C->nz * sizeof(double));

	int m_curr;     // текущая позиция в маске М
	int m_max_pos;  // граница для текущей строки маски M
	int c_curr = 0; // текущая позиция в итоговой матрице C
	for (int i = 0; i < A->V; ++i) {
		priority_queue<spm_iterator> heap;

		// заполнение кучи
		// k - позиция начала строки A->Adjncy[j] в массиве матрицы B
		for (int j = A->Xadj[i]; j < A->Xadj[i+1]; ++j) {
			int k = B->Xadj[A->Adjncy[j]];
			heap.emplace(j, k, B->Adjncy[k]);
		}

		m_curr = M->Xadj[i];
		m_max_pos = M->Xadj[i+1];
		while (!heap.empty()) {
			// достаём итератор с минимальным индексом столбца B
			spm_iterator iter = heap.top();
			heap.pop();

			// если столбец в М меньше минимального текущего столбца в B, ищем первую не меньшую позицию
			while (m_curr < m_max_pos && M->Adjncy[m_curr] < iter.b_col) {
				++m_curr;
				if (C->Adjncy[c_curr] != -1)
					++c_curr;
			}
			// если дошли до конца, выходим
			if (m_curr == m_max_pos)
				break;
			// при совпадении столбцов в M и B умножаем и прибавляем результат
			if (M->Adjncy[m_curr] == iter.b_col) {
				C->Adjncy[c_curr] = A->Adjncy[iter.b_pos];
				C->Eweights[c_curr] += A->Eweights[iter.a_pos] * B->Eweights[iter.b_pos];
			}

			// обработка вставки итератора обратно в кучу
			int b_max_pos = B->Xadj[A->Adjncy[iter.a_pos] + 1];
			// если у итератора столбец B меньше, чем текущий столбец M, увеличиваем, пока он меньше
			++iter.b_pos;
			iter.b_col = B->Adjncy[iter.b_pos];
			int m_curr_pos = M->Adjncy[m_curr];
			while (iter.b_pos < b_max_pos && B->Adjncy[iter.b_pos] < m_curr_pos) {
				++iter.b_pos;
				iter.b_col = B->Adjncy[iter.b_pos];
			}
			// итератор, дошедший до конца строки в B, не вставляется
			if (iter.b_pos < b_max_pos) {
				iter.b_col = B->Adjncy[iter.b_pos];
				heap.push(iter);
			}
		}
		if (C->Adjncy[c_curr] != -1)
			++c_curr;
		C->Xadj[i+1] = c_curr;
	}
	C->nz = C->Xadj[A->V];
}
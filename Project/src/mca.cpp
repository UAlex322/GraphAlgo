#include "mca.h"
#include <stdlib.h>
#include <string.h>

enum {ALLOWED = 0, SET} mca_states;

void mca_init(MCA *accum, size_t len) {
	accum->values = (double*)malloc(len * sizeof(double));
	accum->states =   (char*)malloc(len * sizeof(char));
	accum->len    = len;

	memset(accum->values, 0, accum->len * sizeof(double));
	memset(accum->states, 0, accum->len * sizeof(char));
}

void mca_clear(MCA *accum) {
	memset(accum->values, 0, accum->len*sizeof(double));
	memset(accum->states, 0, accum->len*sizeof(char));
}

void mca_finalize(MCA *accum) {
	free(accum->values);
	free(accum->states);
}

typedef crsGraph crsSpMatrix;
void mspmm_mca(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C) {
	// инициализация C
	C->nz = M->nz;
	C->V  = M->V;
	C->Adjncy   = (int*)malloc(sizeof(int) * C->nz);
	C->Xadj     = (int*)malloc(sizeof(int) * (C->V + 1));
	C->Xadj[0]  = 0;
	C->Eweights = (double*)malloc(sizeof(double) * C->nz);
	memset(C->Eweights, 0, C->nz * sizeof(double));

	// Вычисление максимальной длины строки маски,
	// чтобы единожды зарезервировать достаточно памяти
	int max_len = 0;
	for (int i = 0; i < A->V; ++i)
		if (M->Xadj[i+1] - M->Xadj[i] > max_len)
			max_len = M->Xadj[i+1] - M->Xadj[i];

	MCA accum;
	mca_init(&accum, max_len);

	for (int i = 0; i < A->V; ++i) {
		int m_row_len  = M->Xadj[i+1] - M->Xadj[i];
		int m_row_curr;
		// подсчёт i-й строки матрицы C
		for (int t = A->Xadj[i]; t < A->Xadj[i+1]; ++t) {
			int k = A->Adjncy[t];
			int b_row_curr = B->Xadj[k];
			m_row_curr = M->Xadj[i];
			// прохождение строки и обработка только входящих в маску элементов
			for (int j = 0; j < m_row_len; ++j, ++m_row_curr) {
				// ищем следующий входящий в маску элемент
				while (b_row_curr < B->Xadj[k+1] && B->Adjncy[b_row_curr] < M->Adjncy[m_row_curr])
					++b_row_curr;
				// при нахождении накапливаем значение
				if (b_row_curr < B->Xadj[k+1] && B->Adjncy[b_row_curr] == M->Adjncy[m_row_curr]) {
					accum.states[j] = SET;
					accum.values[j] += A->Eweights[t] * B->Eweights[b_row_curr];
				}
			}
		}
		// заполнение i-й строки матрицы C
		int c_row_curr = C->Xadj[i];
		m_row_curr = M->Xadj[i];
		for (int j = 0; j < m_row_len; ++j, ++m_row_curr) {
			if (accum.states[j] == SET) {
				C->Adjncy[c_row_curr] = M->Adjncy[m_row_curr];
				C->Eweights[c_row_curr++] = accum.values[j];
			}
		}
		C->Xadj[i+1] = c_row_curr;
		// очистка аккумулятора для следующей итерации
		mca_clear(&accum);
	}

	C->nz = C->Xadj[C->V];
	mca_finalize(&accum);
}
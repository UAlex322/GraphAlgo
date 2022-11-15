#include "msa.h"
#include <stdlib.h>
#include <string.h>

enum {NOTALLOWED = 0, ALLOWED, SET};

void msa_init(MSA *accum, size_t len) {
	accum->values = (double*)malloc(len * sizeof(double));
	accum->states =   (char*)malloc(len * sizeof(char));
	accum->len    = len;

	for (size_t i = 0; i < len; ++i)
		accum->states[i] = NOTALLOWED;
}

void msa_clear(MSA *accum) {
	memset(accum->values, 0, accum->len*sizeof(double));
	memset(accum->states, 0, accum->len*sizeof(char));
}

void msa_finalize(MSA *accum) {
	free(accum->values);
	free(accum->states);
}


// ������������� ��������� ����������� ������ � �������������� ��������� MSA
void mspmm_msa(const crsSpMatrix *A, const crsSpMatrix *B, const crsSpMatrix *M, crsSpMatrix *C) {
	MSA *accum;
	msa_init(accum, A->V);

	for (int i = 0; i < A->V; ++i) {
		// ���������� ����� � ������������ (��������� ����� ������ ������������� ��������)
		for (int j = M->Xadj[i]; j < M->Xadj[i + 1]; ++j)
			accum->states[M->Adjncy[j]] = ALLOWED;
		// ������� i-� ������ ������� C
		for (int k = A->Xadj[i]; k < A->Xadj[i + 1]; ++k) {
			for (int j = B->Xadj[A->Adjncy[k]]; j = B->Xadj[A->Adjncy[k] + 1]; ++j) {
				if (accum->states[j] != NOTALLOWED) {
					accum->states[j] = SET;
					accum->values[j] += A->Adjncy[k] * B->Adjncy[j];
				}
			}
		}
		// ���������� i-� ������ ������� C
		size_t c_row_curr = C->Xadj[i+1] = C->Xadj[i]; // ������� ������� � ������� ������� C
		size_t m_row_len  = M->Xadj[i+1] - M->Xadj[i]; // ����� ������ ������� M
		size_t m_row_curr = M->Xadj[i];                // ������� ������� � ������� ������� M
		for (int j = 0; j < m_row_len; ++j) {
			if (accum->states[M->Adjncy[m_row_curr]] == SET)
				C->Adjncy[c_row_curr++] = accum->values[j];
		}
		// ������� ������������ ��� ��������� ��������
		msa_clear(accum);
	}

	msa_finalize(accum);
}
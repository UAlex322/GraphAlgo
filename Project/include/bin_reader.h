#include <cstdlib>

// Формат CRS (по Matrix Market):
// m - число строк в матрице
// n - число столбцов в матрице
// nz - число ненулевых элементов в матрице
// matcode - массив char[4] с информацией о матрице (как в Matrix Market)
// rowstart - массив int[m+1] индексов начала строк матрицы,
//     (m+1)-й элемент ограничивает последнюю строку
// column - массив int[nz] столбцов у каждого элемента матрицы
// values - массив ValType[nz] значений элементов матрицы

// Формат BIN, порядок хранения данных:
// 1) char matcode[4]
// 2) size_t m
// 3) size_t n
// 4) size_t nz
// 5) int* rowstart   (число элементов - m+1)
// 6) int* column     (число элементов - nz)
// 7) ValType* values (число элементов - nz)

// !!!ATTENTION!!!
// 1) Нужно заранее знать тип значений матрицы, 
// чтобы передать '*values' правильного типа. Тип значений матрицы
// лежит в matcode[2], так что его надо считать заранее.
// 'I' - int, 'R' - double.
// 2) В функции выделяется память под массивы - это нужно учитывать,
// чтобы не было утечек памяти.

// Чтение графа CRS в бинарном формате из файла
template <typename ValType>
int read_bin_to_crs(const char *filename, int *m, int *n, int *nz, 
        char *matcode, int *rowstart, int *column, ValType *values) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
        return -1;
    
    size_t m_sizet, n_sizet, nz_sizet;
    
    fread(matcode, 1, 1, fp);
    fread(matcode + 1, 1, 1, fp);
    fread(matcode + 2, 1, 1, fp);
    fread(matcode + 3, 1, 1, fp);
    fread(&m_sizet, sizeof(size_t), 1, fp);
    fread(&n_sizet, sizeof(size_t), 1, fp);
    fread(&nz_sizet, sizeof(size_t), 1, fp);
    
    *m  = (int)m_sizet;
    *n  = (int)n_sizet;
    *nz = (int)nz_sizet;
    rowstart = new int[*m+1];
    column = new int[*nz];
    values = new ValType[*nz];
    
    fread(rowstart, sizeof(int), *m + 1, fp);
    fread(column, sizeof(int), *nz, fp);
    fread(values, sizeof(ValType), *nz, fp);
    
    fclose(fp);
    return 0;
}

// Запись графа CRS в память в бинарном формате
template <typename ValType>
int write_crs_to_bin(const char *filename, int *m, int *n, int *nz,
    char *matcode, int *rowstart, int *column, ValType *values) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
        return -1;
    
    size_t m_sizet = (size_t)*m, 
           n_sizet = (size_t)*n, 
           nz_sizet = (size_t)*nz;
    
    fwrite(matcode, 1, 1, fp);
    fwrite(matcode + 1, 1, 1, fp);
    fwrite(matcode + 2, 1, 1, fp);
    fwrite(matcode + 3, 1, 1, fp);
    fwrite(&m_sizet, sizeof(size_t), 1, fp);
    fwrite(&n_sizet, sizeof(size_t), 1, fp);
    fwrite(&nz_sizet, sizeof(size_t), 1, fp);
    fwrite(rowstart, sizeof(int), *m+1, fp);
    fwrite(column, sizeof(int), *nz, fp);
    fwrite(values, sizeof(ValType), *nz, fp);

    fclose(fp);
    return 0;
}
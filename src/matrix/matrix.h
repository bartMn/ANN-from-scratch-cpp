#ifndef MATRIX_H
#define MATRIX_H

class Matrix{
    public:
        Matrix(int r, int c): rows(r), columns(c) {}
        void test();
        int get_rows_num();
        int get_columns_num();
        

    private:
        int rows;
        int columns;
};


#endif
#ifndef MATRIX_H
#define MATRIX_H

class Matrix{
    public:
        Matrix(int r, int c): rows(r), columns(c) {}
        void test();
        

    private:
        int rows;
        int columns;
};


#endif
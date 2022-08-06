STREQ(s1, s2) returns FALSE iff strings s1 and s2 are equal.
scopy(s) returns a copy of the string s.
error(format, arg1...) prints a message and causes the program to exit.
mallocOrDie(nbytes) calls malloc, dying with an error message if the memory cannot be obtained.

CLEAR_BIT_VECTOR(bv) resets an entire bit vector bv to all zeros,
BIT_SET(i, bv) sets the ith bit of bv to one, and
BIT_CLEAR(i, bv) sets the ith bit of bv to zero.

double QuadraticForm(Vector V, Matrix M); computes the quantity V MV, where
    the prime denotes the transpose operation.

Matrix SliceMatrix(Matrix m, BitVector rowmask, BitVector colmask);
    creates a new matrix, consisting only of those rows and columns in m whose
    corresponding bits are set it rowmask and colmask, respectively.

Matrix DeSliceMatrix(Matrix m, double fill, BitVector rowmask; BitVector colmask; Matrix result);
    first sets every element in result to fill, and then, every element in result
    whose row number is on in rowmask and whose column numberis on in colmask,
    is set from the corresponding element in the input matrix m, which is smaller
    than r.
    The result of SliceMatrix(DeSliceMatrix(m, fill, rowmask, colmask, result), rowmask, colmask)
    is a copy of m, given legal values for all parameters.

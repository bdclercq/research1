import numpy as np


def QuadraticForm(vec, mat):
    v_p = np.transpose(vec)
    mv = np.dot(mat, vec)
    return np.inner(v_p, mv)


def SliceMatrix(mat, rowmask, colmask):
    r = [i for i in range(len(rowmask)) if rowmask[i]]
    to_ret = mat[r]
    c = [i for i in range(len(colmask)) if colmask[i]]
    return to_ret[:, c]


def DeSliceMatrix(m, fill, rowmask, colmask, result):
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = fill

    r = [i for i in range(len(rowmask)) if rowmask[i]]
    c = [i for i in range(len(colmask)) if colmask[i]]

    for i in r:
        for j in c:
            result[i][j] = m[i][j]

    return result


def OutputVector(vector):
    # Return 1 line writeable version of the vector
    return ["".join(str(i)) for i in vector]


def OutputMatrix(matrix):
    # Return 1 line writeable version of the matrix
    line = ""
    for row in matrix:
        for col in row:
            line.join(str(col))
        line.join(";")
    return line


def InputVector(input):
    pass


def InputMatrix(input):
    pass

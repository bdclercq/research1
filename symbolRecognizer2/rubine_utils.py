import numpy as np


def QuadraticForm(vec, mat):
    v_p = np.transpose(vec)
    mv = np.dot(mat, vec)
    return np.inner(v_p, mv)


def SliceMatrix(mat, rowmask, colmask):
    assert len(mat) == len(rowmask)
    # print("Input matrix for SliceMatrix:\n", mat)
    r = [i for i in range(len(rowmask)) if rowmask[i]]
    # print("Rows to keep: ", r)
    to_ret = []
    for i in r:
        # print(mat[i])
        to_ret.append(mat[i])
    # print("Intermediate result: ", to_ret)
    c = [i for i in range(len(colmask)) if colmask[i]]
    # print("Columns to keep: ", c)
    for j in range(len(to_ret)):
        # print(row)
        to_ret[j] = [to_ret[j][i] for i in c]
    return to_ret


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
    return "".join(str(vector))


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

"""
These functions where probably implemented by Rubine in a separate file, but as this file was not included
the implementations have been based on the description he offered in his thesis.
"""
import numpy as np


def QuadraticForm(vec, mat):
    """
    Computes V' MV where V' denotes the transpose operation.
    :param vec: a vector
    :param mat: a matrix
    :return: V' MV
    """
    v_p = np.transpose(vec)
    mv = np.dot(mat, vec)
    return np.inner(v_p, mv)


def SliceMatrix(mat, rowmask, colmask):
    """
    Creates a new matrix, consisting only of those rows and columns in mat whose corresponding
        bits are set it rowmask and colmask, respectively.
    :param mat: matrix to slice
    :param rowmask: rows to mask
    :param colmask: columns to mask
    :return: sliced matrix
    """
    assert len(mat) == len(rowmask)
    r = [i for i in range(len(rowmask)) if rowmask[i]]
    to_ret = []
    for i in r:
        to_ret.append(mat[i])
    c = [i for i in range(len(colmask)) if colmask[i]]
    for j in range(len(to_ret)):
        to_ret[j] = [to_ret[j][i] for i in c]
    return to_ret


def DeSliceMatrix(m, fill, rowmask, colmask, result):
    """
    First sets every element in result to fill, and then, every element in result whose row number is on
    in rowmask and whose column number is on in colmask, is set from the corresponding element in the
    input matrix m, which is smaller than r.
    :param m: original input matrix
    :param fill: default value to fill resulting matrix with
    :param rowmask: mask for rows to keep
    :param colmask: mask for columns to keep
    :param result: result matrix
    :return: returns the resulting matrix
    """
    print(result)
    print(m)
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = np.zeros(result[i][j].shape)

    r = [i for i in range(len(rowmask)) if rowmask[i]]
    c = [i for i in range(len(colmask)) if colmask[i]]
    for i in r:
        for j in c:
            print(result[i][j], ", ", m[i][j])
            result[i][j] = m[i][j]

    return result


def OutputVector(vector):
    """
    Return 1 line writeable version of the vector
    :param vector: vector to translate
    :return: string version of vector
    """
    line = ""
    for v in vector:
        line += "{0} ".format(v)
    return line


def OutputMatrix(matrix):
    """
    Return 1 line writeable version of the matrix
    :param matrix: matrix to translate
    :return: string version of the matrix
    """
    line = ""
    for row in matrix:
        for col in row:
            line += "{0} ".format(col)
        line += ";"
    return line


def InputVector(input):
    line = input.split(" ")
    to_ret = []
    for i in range(len(line)-1):
        val = 0
        try:
            val = float(line[i])
        except:
            try:
                l = line[i].strip('[]')
                if float(l) == 0.0:
                    val = 0.0
                else:
                    val = np.matrix(float(l))
            except Exception as e:
                raise Exception('Cannot read vector: {0}'.format(e))
        to_ret.append(val)
    return to_ret


def InputMatrix(input):
    lines = input.split(";")
    m = []
    for l in range(len(lines)-1):
        m.append(InputVector(lines[l]))
    return m

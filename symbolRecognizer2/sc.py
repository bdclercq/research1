"""
create single path classifiers from feature vectors of examples,
as well as classifying example feature vectors.
"""
import math
import numpy as np
import rubine_utils as ru

MAXSCLASSES = 100
EPS = math.pow(10, -6)


class sClassDope:
    """
    per gesture class information within a classifier
    """
    def __init__(self):
        self.name = ""
        self.number = -1
        self.nexamples = 0
        self.average =[]
        self.sumcov = [[]]


class sClassifier:
    """
    a classifier
    """
    # static variables
    lastsc = 0
    lastscd = 0
    space = []

    def __init__(self):
        self.nfeatures = -1
        self.nclasses = 0
        self.classdope = []
        self.cnst = []
        self.w = []
        self.invavgcov = [[]]

    def sClassNameLookup(self, classname):
        if lastsc == self and (lastscd.name == classname):
            return lastscd
        for i in range(self.nclasses):
            scd = self.classdope[i]
            if scd.name == classname:
                lastsc = self
                lastscd = scd
                return
        return 0

    def sAddClass(self, classname):
        scd = sClassDope()
        self.classdope[self.nclasses] = scd
        scd.name = classname
        scd.number = self.nclasses
        scd.nexamples = 0
        scd.sumcov = [[]]
        self.nclasses += 1
        return scd

    def sAddExample(self, classname, y):
        nfv = [0 for i in range(50)]

        scd = self.sClassNameLookup(self, classname)
        if scd == 0:
            scd = self.sAddClass(self, classname)

        if self.nfeatures == -1:
            self.nfeatures = len(y)

        if (scd.nexamples == 0):
            scd.average = [0 for i in range(self.nfeatures)]
            scd.sumcov = [[0 for i in range(self.nfeatures)] for j in range(self.nfeatures)]

        if (self.nfeatures != len(y)):
            print(y, " sAddExample: funny vector nrows!=%d", self.nfeatures)
            return

        scd.nexamples += 1
        nm1on = (scd.nexamples-1.0) / scd.nexamples
        recipn = 1.0 / scd.nexamples

        for i in range(self.nfeatures):
            nfv[i] = y[i] - scd.average[i]

        for i in range(self.nfeatures):
            for j in range(i, self.nfeatures):
                scd.sumcov[i][j] += nm1on * nfv[i] * nfv[j]

        for i in range(self.nfeatures):
            scd.average[i] = nm1on * scd.average[i] + recipn * y[i]


    def sDoneAdding(self):
        if self.nclasses == 0:
            raise ("sDoneAdding: No classes\n")

        avgcov = [[0 for i in range(self.nfeatures)] for j in range(self.nfeatures)]
        ne = 0
        for c in range(self.nclasses):
            scd = self.classdope[c]
            ne += scd.nexamples
            s = scd.sumcov
            for i in range(self.nfeatures):
                for j in range(i, self.nfeatures):
                    avgcov[i][j] += s[i][j]

        denom = ne - self.nclasses
        if denom <= 0:
            print("no examples, denom=%d\n", denom)
            return

        oneoverdenom = 1.0 / denom
        for i in range(self.nfeatures):
            for j in range(i, self.nfeatures):
                avgcov[j][i] = oneoverdenom
                avgcov[i][j] = oneoverdenom

        self.invavgcov = [[None for i in range(self.nfeatures)] for j in range(self.nfeatures)]
        det = np.linalg.det(np.matrix(avgcov).I)
        if math.fabs(det) <= EPS:
            self.FixClassifier(avgcov)

        self.w = [None for i in range(self.nclasses)]
        self.cnst = [None for i in range(self.nclasses)]
        for c in range(self.nclasses):
            scd = self.classdope[c]
            self.w[c] = [None for i in range(self.nfeatures)]
            np.dot(scd.average, self.invavgcov, self.w[c])
            self.cnst[c] = -0.5 * np.inner(self.w[c], scd.average)

        return

    def sClassify(self, fv):
        return self.sClassifyAD(fv, None, None)

    def sClassifyAD(self, fv, ap, dp):
        disc = [None for i in range(MAXSCLASSES)]

        if not self.w:
            raise ("sClassifyAD: %x no trained classifier", self)

        for i in range(self.nclasses):
            disc[i] = np.inner(self.w[i], fv) + self.cnst[i]

        maxclass = 0
        for i in range(self.nclasses):
            if disc[i] > disc[maxclass]:
                maxclass = i

        scd = self.classdope[maxclass]

        if ap:
            for denom, i in range(self.nclasses):
                d = disc[i] - disc[maxclass]
                if d > -7.0:
                    denom += math.exp(d)
                ap = 1.0 / denom


        if dp:
            dp = self.MahalanobisDistance(fv, scd.average, self.invavgcov)

        return scd

    def MahalanobisDistance(self, v, u, sigma):
        if not space or len(space) != len(v):
            space = [None for i in range(len(v))]

        for i in range(len(v)):
            space[i] = v[i] - u[i]

        result = ru.QuadraticForm(space, sigma)
        return result

    def FixClassifier(self, avgcov):
        bv = [0 for i in range(self.nfeatures)]

        for i in range(self.nfeatures):
            bv[i] = 1
            m = ru.SliceMatrix(avgcov, bv, bv)
            det = np.linalg.det(np.matrix(m).I)
            if math.fabs(det) <= EPS:
                bv[i] = 0

        m = ru.SliceMatrix(avgcov, bv, bv)
        r = np.matrix(m).I
        det = np.linalg.det(r)
        if math.fabs(det) <= EPS:
            raise Exception("Can't fix classifier!")
        ru.DeSliceMatrix(r, 0.0, bv, bv, self.invavgcov)

    def write(self, outfile):
        file = open(outfile, "w")
        file.write("%d classes\n".format(self.nclasses))
        for i in range(self.nclasses):
            scd = self.classdope[i]
            file.write("%s\n".format(scd.name))

        for i in range(self.nclasses):
            scd = self.classdope[i]
            file.write(ru.OutputVector(scd.average))
            file.write(ru.OutputVector(self.w[i]))

        file.write(ru.OutputVector(self.cnst))
        file.write(ru.OutputMatrix(self.invavgcov))

    def read(self, infile):
        print("Reading classifier ")

        n = None

        file = open(infile, 'r', 100)
        buf = file.readline()
        try:
            n = int(buf)
        except:
            raise Exception("sRead 1")
        print("%d classes ", n)
        for i in range(n):
            buf = file.readline()
            scd = self.sAddClass(buf)
            scd.name = buf
            print("%s ", scd.name)

        self.w = [None for i in range(self.nclasses)]
        for i in range(self.nclasses):
            scd = self.classdope[i]
            scd.average = ru.InputVector(file.readline())
            self.w[i] = ru.InputVector(file.readline())

        self.cnst = ru.InputVector(file.readline())
        self.invavgcov = ru.InputMatrix(file.readline())
        print("\n")

    def sDistances(self, nclosest):
        d = [[None for i in range(self.nclasses)] for j in range(self.nclasses)]
        min, max = 0, 0
        n, mi, mj = 0, 0, 0

        print("----------\n");
        print("%d closest pairs of classes\n", nclosest)
        for i in range(len(d)):
            for j in range(len(d[i])):
                d[i][j] = self.MahalanobisDistance(
                            self.classdope[i].average,
                            self.classdope[j].average,
                            self.invavgcov)
        if d[i][j] > max:
            max = d[i][j]

        for n in range(nclosest):
            min = max
            mi, mj = -1, -1
            for i in range(len(d)):
                for j in range(len(d[i])):
                    if d[i][j] < min:
                        mi, mj = i, j
                        min = d[mi][mj]
            if mi == 1:
                break

            print("%2d) %10.10s to %10.10s d=%g nstd=%g\n",
                    n,
                    self.classdope[mi].name,
                    self.classdope[mj].name,
                    d[mi][mj],
                    math.sqrt(d[mi][mj]))

            d[mi][mj] = max + 1

        print("----------\n")

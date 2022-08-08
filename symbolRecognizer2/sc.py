"""
Create single path classifiers from feature vectors of examples,
as well as classifying example feature vectors.
"""
import math
import numpy as np
import rubine_utils as ru

MAXSCLASSES = 100
EPS = math.pow(10, -6)


class sClassDope:
    """
    Per gesture class information within a classifier
    """

    def __init__(self):
        self.name = ""
        self.number = -1
        self.nexamples = 0
        self.average = []
        self.sumcov = [[]]


class sClassifier:
    """
    A classifier
    """

    def __init__(self):
        self.nfeatures = -1
        self.nclasses = 0
        self.classdope = []
        self.cnst = []
        self.w = []
        self.invavgcov = [[]]
        self.lastsc = self
        self.lastscd = sClassDope()
        self.space = []

    def sClassNameLookup(self, classname):
        """
        Given a string name of a class, return its per class information.
        :param classname: name of the class
        :return: sClassDope if it exists, 0 else
        """
        # Quick check for last class name
        if self.lastsc == self and (self.lastscd.name == classname):
            return self.lastscd

        # Linear search through all classes for name
        for i in range(self.nclasses):
            scd = self.classdope[i]
            if scd.name == classname:
                self.lastsc = self
                self.lastscd = scd
                return self.lastscd
        return 0

    def sAddClass(self, classname):
        """
        Add a new gesture class to a classifier.
        :param classname: name of the class
        :return: sClassDope that has been added
        """
        scd = sClassDope()
        # Do not move this append further down or recognition will fail
        self.classdope.append(scd)
        scd.name = classname
        scd.number = self.nclasses
        scd.nexamples = 0
        scd.sumcov = [[]]
        self.nclasses += 1
        return scd

    def sAddExample(self, classname, y):
        """
        Add a new training example to a classifier.
        :param classname: name of the class
        :param y: feature vector
        :return: void
        """
        nfv = [0 for i in range(50)]

        # Search for the classname in existing scd's, add new if not found
        scd = self.sClassNameLookup(classname)
        if scd == 0:
            scd = self.sAddClass(classname)

        # If this is the first time we add something to the classifier, set the number of features
        if self.nfeatures == -1:
            self.nfeatures = len(y)

        # If this is the first example of the sClassDope, set some values
        if scd.nexamples == 0:
            scd.average = [0 for i in range(self.nfeatures)]
            scd.sumcov = [[0 for i in range(self.nfeatures)] for j in range(self.nfeatures)]

        if self.nfeatures != len(y):
            print(y, " sAddExample: funny vector nrows!={0}".format(self.nfeatures))
            return

        scd.nexamples += 1
        nm1on = (scd.nexamples - 1.0) / scd.nexamples
        recipn = 1.0 / scd.nexamples

        # Incrementally update covariance matrix
        for i in range(self.nfeatures):
            nfv[i] = y[i] - scd.average[i]

        # Only upper triangular part computed
        for i in range(self.nfeatures):
            for j in range(i, self.nfeatures):
                scd.sumcov[i][j] += nm1on * nfv[i] * nfv[j]

        # Incrementally update mean vector
        for i in range(self.nfeatures):
            scd.average[i] = nm1on * scd.average[i] + recipn * y[i]

    def sDoneAdding(self):
        """
        Run the training algorithm on the classifier.
        :return: void
        """
        if self.nclasses == 0:
            raise Exception("sDoneAdding: No classes\n")

        # Given covariance matrices for each class ( number of examples	- 1),
        # compute the average (common) covariance matrix
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
            print("no examples, denom={0}\n".format(denom))
            return

        oneoverdenom = 1.0 / denom
        for i in range(self.nfeatures):
            for j in range(i, self.nfeatures):
                avgcov[j][i] = oneoverdenom
                avgcov[i][j] = oneoverdenom

        # Invert the avg covariance matrix
        self.invavgcov = [[None for i in range(self.nfeatures)] for j in range(self.nfeatures)]
        det = 0
        try:
            avgcov_inv = np.matrix(avgcov).I
            det = np.linalg.det(avgcov_inv)
        except:
            pass
        if math.fabs(det) <= EPS:
            self.FixClassifier(avgcov)

        # Now compute discrimination functions
        self.w = [[] for i in range(self.nclasses)]
        self.cnst = [None for i in range(self.nclasses)]
        for c in range(self.nclasses):
            scd = self.classdope[c]
            self.w[c] = np.dot(scd.average, self.invavgcov)
            self.cnst[c] = -0.5 * np.inner(self.w[c], scd.average)
        return

    def sClassify(self, fv):
        """
        Classify a feature vector.
        :param fv: feature vector
        :return: void
        """
        return self.sClassifyAD(fv, 0, 0)

    def sClassifyAD(self, fv, ap=1, dp=1):
        """
        Classify a feature vector, possibly computing rejection metrics.
        :param fv: feature vector
        :param ap: value that indicates whether to calculate the probability of unambiguous classification
        :param dp: value that indicates whether to calculate the distance from the class mean
        :return: sClassDope, ap, dp
        """
        disc = [None for i in range(MAXSCLASSES)]

        if not self.w:
            raise Exception("sClassifyAD: {0} no trained classifier".format(self))

        for i in range(self.nclasses):
            disc[i] = np.inner(self.w[i], fv) + self.cnst[i]

        maxclass = 0
        for i in range(1, self.nclasses):
            if disc[i] > disc[maxclass]:
                maxclass = i

        scd = self.classdope[maxclass]

        if ap:
            # Calculate probability of non ambiguity
            denom = 0
            for i in range(self.nclasses):
                d = disc[i] - disc[maxclass]
                print(d)
                # Quick check to avoid computing negligible term
                if d > -7.0:
                    denom += math.exp(d)
                # Handle the case where denom remains 0, as happened when drawing circles.
                # According to the IEEE 754 Standard, division by zero should result in INF.
                if denom == 0:
                    ap = 1.0    # math.inf, or a probability of 100%
                else:
                    ap = 1.0 / denom

        if dp:
            # Calculate distance to mean of chosen class
            dp = self.MahalanobisDistance(fv, scd.average, self.invavgcov)

        return scd, ap, dp

    def MahalanobisDistance(self, v, u, sigma):
        """
        Compute the Mahalanobis distance between two vectors v and u.
        :param v: feature vector
        :param u: average features vector of class
        :param sigma: inverse covariance matrix of class
        :return: distance
        """
        if not self.space or len(self.space) != len(v):
            self.space = [None for i in range(len(v))]

        for i in range(len(v)):
            self.space[i] = v[i] - u[i]

        result = ru.QuadraticForm(self.space, sigma)
        return result

    def FixClassifier(self, avgcov):
        """
        Handle the case of a singular average covariance matrix by removing features.
        :param avgcov: singular average covariance matrix
        :return: void
        """
        bv = [0 for i in range(self.nfeatures)]
        # Just add the features one by one, discarding any that cause the matrix to be non invertible
        for i in range(self.nfeatures):
            bv[i] = 1
            m = ru.SliceMatrix(avgcov, bv, bv)
            det = 0
            try:
                det = np.linalg.det(np.matrix(m).I)
            except:
                pass
            if math.fabs(det) <= EPS:
                bv[i] = 0

        m = ru.SliceMatrix(avgcov, bv, bv)
        r = np.matrix(m).I
        det = 0
        try:
            det = np.linalg.det(r)
        except:
            pass
        if math.fabs(det) <= EPS:
            raise Exception("Can't fix classifier!")
        self.invavgcov = ru.DeSliceMatrix(r, 0.0, bv, bv, self.invavgcov)

    def write(self, outfile):
        """
        Write a classifier to a file.
        :param outfile: name of the output file
        :return: void
        """
        file = open(outfile, "w")
        file.write("{0} classes\n".format(self.nclasses))
        for i in range(self.nclasses):
            scd = self.classdope[i]
            file.write("{0}\n".format(scd.name))

        for i in range(self.nclasses):
            scd = self.classdope[i]
            print("average:")
            file.write("{0}\n".format(ru.OutputVector(scd.average)))
            print("w[i]:")
            file.write("{0}\n".format(ru.OutputVector(self.w[i])))

        print("cnst:")
        file.write("{0}\n".format(ru.OutputVector(self.cnst)))
        print("invavgcov:")
        file.write("{0}\n".format(ru.OutputMatrix(self.invavgcov)))
        file.close()

    def read(self, infile):
        """
        Read a classifier from a file.
        :param infile: name of the input file
        :return: void
        """
        print("Reading classifier ")

        n = None

        file = open(infile, 'r')
        buf = file.readline()
        try:
            n = int(buf.split(" ")[0])
        except:
            raise Exception("sRead 1")
        print("{0} classes ".format(n))
        for i in range(n):
            buf = file.readline()
            scd = self.sAddClass(buf)
            scd.name = buf
            print("{0}".format(scd.name))

        self.w = [None for i in range(self.nclasses)]
        for i in range(self.nclasses):
            scd = self.classdope[i]
            scd.average = ru.InputVector(file.readline())
            self.w[i] = ru.InputVector(file.readline())

        self.cnst = ru.InputVector(file.readline())
        self.invavgcov = ru.InputMatrix(file.readline())
        print("\n")
        file.close()

    def sDistances(self, nclosest):
        """
        compute pairwise distances between classes, and print the closest ones,
            as a clue as to which gesture classes are confusable.
        :param nclosest: closest pairs of classes
        :return:
        """
        d = [[None for i in range(self.nclasses)] for j in range(self.nclasses)]
        min, max = 0, 0

        print("----------\n")
        print("{0} closest pairs of classes\n".format(nclosest))
        for i in range(len(d)):
            for j in range(i+1, len(d[i])):
                d[i][j] = self.MahalanobisDistance(
                    self.classdope[i].average,
                    self.classdope[j].average,
                    self.invavgcov)
        if d[i][j] > max:
            max = d[i][j]

        for n in range(1, len(nclosest)):
            min = max
            mi, mj = -1, -1
            for i in range(len(d)):
                for j in range(i+1, len(d[i])):
                    if d[i][j] < min:
                        mi, mj = i, j
                        min = d[mi][mj]
            if mi == 1:
                break

            print("{0}) {1} to {2} d= {3} nstd={4}\n".format(
                  n,
                  self.classdope[mi].name,
                  self.classdope[mj].name,
                  d[mi][mj],
                  math.sqrt(d[mi][mj])))

            d[mi][mj] = max + 1

        print("----------\n")

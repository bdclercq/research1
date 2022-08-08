"""
Create a feature vector, useful for gesture classification,
from a sequence of points (e.g. mouse points).
"""
"""
/* indices into the feature Vector returned by FvCalc */

#define PF_INIT_COS	0 /* initial angle (cos) */
#define PF_INIT_SIN	1 /* initial angle (sin)  */
#define PF_BB_LEN	2 /* length of bounding box diagonal */
#define PF_BB_TH	3 /* angle of bounding box diagonal  */
#define PF_SE_LEN	4 /* length between start and end points */
#define PF_SE_COS	5 /* cos of angle between start and end points */
#define PF_SE_SIN	6 /* sin of angle between start and end points */
#define PF_LEN	7 /* arc length of path */
#define PF_TH	8 /* total angle traversed */
#define PF_ATH		9 /* sum of abs vals of angles traversed */
#define PF_SQTH	10 /* sum of squares of angles traversed */
"""
import math

PF_INIT_COS = 0
PF_INIT_SIN = 1
PF_BB_LEN = 2
PF_BB_TH = 3
PF_SE_LEN = 4
PF_SE_COS = 5
PF_SE_SIN = 6
PF_LEN = 7
PF_TH = 8
PF_ATH = 9
PF_SQTH = 10
PF_DUR = 11
PF_MAXV = 12

NFEATURES = 13
dist_sq_threshold = (3 * 3)
se_th_rolloff = (4 * 4)
EPS = math.pow(10, -4)


class FV:
    """
    Structure which holds intermediate results during feature vector calculation
    """
    def __init__(self):
        # The following are used in calculating the features
        self.startx = 0
        self.starty = 0
        self.starttime = 0

        # Initial angle to x axis, these are set after a few points and then left alone
        self.initial_sin = 0.0
        self.initial_cos = 0.0

        # These are updated incrementally upon every point
        self.npoints = 0
        # Differences: endx prevx, endy prevy
        self.dx2 = 0
        self.dy2 = 0
        self.magsq2 = self.dx2 * self.dx2 + self.dy2 * self.dy2
        # Last point added
        self.endx = 0
        self.endy = 0
        self.endtime = 0

        # Bounding box
        self.minx = 0
        self.maxx = 0
        self.miny = 0
        self.maxy = 0

        # Total length and rotation (in radians)
        self.path_r = 0
        self.path_th = 0
        # Sum of absolute values of path angles
        self.abs_th = 0
        # Sum of squares of path angles
        self.sharpness = 0
        # Maximum velocity
        self.maxv = 0

        # Actual feature vector
        self.y = [0.0 for i in range(NFEATURES)]

    def AddPoint(self, x, y, t):
        """
        Update a FV to reflect a new input point
        :param x: x-coordinate of the new point
        :param y: y-coordinate of the new point
        :param t: time values
        :return: void
        """
        self.npoints += 1

        # first point, initialize some vars
        if self.npoints == 1:
            self.starttime = t
            self.endtime = t
            self.startx = x
            self.endx = x
            self.minx = x
            self.maxx = x
            self.starty = y
            self.endy = y
            self.miny = y
            self.maxy = y
            return
        dx1 = x - self.endx
        dy1 = y - self.endy

        magsq1 = dx1 * dx1 + dy1 * dy1

        if magsq1 <= dist_sq_threshold:
            self.npoints -= 1
            return  # ignore a point close to the last point
        # Update some internal values if needed
        if x < self.minx:
            self.minx = x
        if x > self.maxx:
            self.maxx = x
        if y < self.miny:
            self.miny = y
        if y > self.maxy:
            self.maxy = y

        lasttime = self.endtime

        self.endtime = t

        d = math.sqrt(magsq1)
        self.path_r += d  # update path length feature

        # calculate initial theta when the third point is seen
        if self.npoints == 3:
            dx = x - self.startx
            dy = y - self.starty
            magsq = dx * dx + dy * dy
            if magsq > dist_sq_threshold:
                # find angle w.r.t.positive x axis e.g.(1, 0)
                recip = 1.0 / math.sqrt(magsq)
                self.initial_cos = dx * recip
                self.initial_sin = dy * recip

        if self.npoints >= 3:
            # / * update angle based features * /
            th = math.atan2(dx1 * self.dy2 - self.dx2 * dy1,
                            dx1 * self.dx2 + dy1 * self.dy2)
            absth = math.atan2(dx1 * self.dy2 - self.dx2 * dy1,
                               dx1 * self.dx2 + dy1 * self.dy2)
            if absth < 0:
                absth = -absth
            self.path_th += th
            self.abs_th += absth
            self.sharpness += th * th

            # Compute max velocity
            if self.endtime > lasttime and (d / (self.endtime - lasttime)) > self.maxv:
                self.maxv = d / (self.endtime - lasttime)

        # prepare for next iteration
        self.endx = x
        self.endy = y
        self.dx2 = dx1
        self.dy2 = dy1
        self.magsq2 = magsq1

        return

    def FvCalc(self):
        """
        Calculate and return a feature vector
        :return: feature vector
        """
        if self.npoints <= 1:
            return self.y   # A feature vector of all zeros
        self.y[PF_INIT_COS] = self.initial_cos
        self.y[PF_INIT_SIN] = self.initial_sin

        # Compute the length of the bounding box diagonal
        bblen = math.hypot(self.maxx - self.minx, self.maxy - self.miny)

        self.y[PF_BB_LEN] = bblen

        # The bounding box angle defaults to 0 for small gestures
        if bblen * bblen > dist_sq_threshold:
            self.y[PF_BB_TH] = math.atan2(self.maxy - self.miny, self.maxx - self.minx)

        # Compute the length and angle between the first and last points
        selen = math.hypot(self.endx - self.startx, self.endy - self.starty);
        self.y[PF_SE_LEN] = selen

        # When the first and last points are very close,
        # the angle features are muted so that they satisfy the stability criterion
        factor = selen * selen / se_th_rolloff
        if factor > 1.0:
            factor = 1.0

        if selen > EPS:
            factor /= selen
        else:
            factor = 0
        self.y[PF_SE_COS] = (self.endx - self.startx) * factor
        self.y[PF_SE_SIN] = (self.endy - self.starty) * factor

        # The remaining features have already been computed
        self.y[PF_LEN] = self.path_r
        self.y[PF_TH] = self.path_th
        self.y[PF_ATH] = self.abs_th
        self.y[PF_SQTH] = self.sharpness

        self.y[PF_DUR] = (self.endtime - self.starttime) * .01

        self.y[PF_MAXV] = self.maxv * 10000

        return self.y

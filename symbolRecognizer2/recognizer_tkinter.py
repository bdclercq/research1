import math
import time
from tkinter import *
from datetime import datetime, timedelta
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.stats import linregress
import numpy as np

dim_x = 950
dim_y = 750


class Recognizer(Frame):
    def __init__(self):
        super().__init__()

        self.time = datetime.now()
        self.delta = timedelta()
        self.strokes = []
        self.points = []
        self.gestures = []
        # Create canvas on window
        self.canvas = Canvas(width=dim_x, height=dim_y)
        self.canvas.pack(expand=1)
        self.timeout = "idle"

        self.done = False
        self.moving = True
        self.drawing = False
        self.update = False
        self.p = True

        self.init_canvas()

    def init_canvas(self):
        # Bind actions
        # Only move mouse, not button pressed
        self.canvas.bind('<Motion>', self.motion)
        # Click left mouse button
        self.canvas.bind('<Button-1>',
                         self.start_timer)
        # Release left mouse button
        self.canvas.bind('<ButtonRelease-1>', self.save_stroke)
        # Move the mouse while left button pressed
        self.canvas.bind('<B1-Motion>', self.stroke)
        self.canvas.pack()

    def motion(self, event):
        self.done = False
        self.moving = True
        self.drawing = False

    def start_timer(self, event):
        """
        Start the timer for the current gesture
        :param event: tkinter event
        :return: void
        """
        self.time = datetime.now()

    def stroke(self, event):
        """
        Collect points while the left mouse button is pressed.
        :param event: tkinter event from which we derive the current position on the screen
        :return: void
        """
        self.drawing = True
        self.done = False
        if self.p:
            self.points.append((event.x, event.y))
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.canvas.create_oval(x1, y1, x2, y2)
        self.p = not self.p

    # Left mouse button is released, save stroke
    def save_stroke(self, event):
        """
        Save a stroke if the left mouse button is released.
        :param event: tkinter event
        :return: void
        """
        # We are ready to update our collection of strokes
        self.update = True
        self.done = True
        self.moving = False
        self.strokes.append(self.points)
        # Arrange the coordinates in a list
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        # Set the timer as follows: 500ms as the basis, + 4 times the largest distance traveled
        self.timeout = 500 + 4 * max((max(x) - min(x)), (max(y) - min(y)))
        # Prepare for a possible next stroke
        self.points = []
        end = datetime.now()
        self.time = end
        # If the timer expires: process strokes as a single gesture
        self.canvas.after(self.timeout, self.process_strokes)

    def calc_perimeter(self, points):
        """
        Calculate the perimeter of a sketch defined by points.
        :param points: aray of points [(xi, yi)]
        :return: perimeter of the sketch defined by points
        """
        # Default perimeter is 0
        perimeter = 0
        for i in range(len(points) - 1):
            # Update perimeter
            if i != len(points) - 1:
                perimeter += math.pow(
                    math.pow(points[i][0] - points[i + 1][0], 2) + math.pow(points[i][1] - points[i + 1][1], 2), 0.5)
            else:
                perimeter += math.pow(
                    math.pow(points[i][0] - points[-1][0], 2) + math.pow(points[i][1] - points[-1][1], 2), 0.5)
        return perimeter

    def calc_area(self, points):
        """
        Function implements Shoelace method to calculate area of a polygon.
        Source of code: https://stackoverflow.com/a/53864271/11482532
        :param points: aray of points [(xi, yi)]
        :return: area of the polygon defined by the given points
        """
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        x_ = x - x.mean()
        y_ = y - y.mean()
        # Correction term is introduced to allow for more performant code.
        # For complete reasoning and source of the code see https://stackoverflow.com/a/53864271/11482532
        correction = x_[-1] * y_[0] - y_[-1] * x_[0]
        area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        return 0.5 * np.abs(area + correction)

    def area_ratio_filter(self, x, y, Ac):
        """
        Filter that looks at the ratio of the area defined by the convex hull and the bounding box.
        :param x: X coordinates of the convex hull, smallest and largest value are used to determine bounding box
        :param y: y coordinates of the convex hull, smallest and largest value are used to determine bounding box
        :param Ac: area of the convex hull
        :return: Return ratio.
        """
        bbox_points = np.array([(np.min(x), np.min(y)), (np.min(x), np.max(y)),
                                (np.max(x), np.max(y)), (np.max(x), np.min(y))])
        Ar = self.calc_area(bbox_points)
        ratio = Ac / Ar
        return ratio

    def triangle_diamond_filter(self, segments, points):
        """
        Distinguish between triangle or diamond shapes.
        :param segments: segments of the current shape
        :param points: points of the sketch
        :return: number of corners found
        """
        # Find corners of sketch
        # Split x, y in separate lines
        corners = []
        seg0 = segments[0]
        x0 = [p[0] for p in seg0]
        y0 = [p[1] for p in seg0]
        slope0, intercept0, r_value, p_value, std_err = linregress(x0, y0)
        slope = slope0
        for i in range(len(segments)):
            if i == len(segments) - 1:
                # Compare last with first
                next_seg = segments[i]
                next_x = [p[0] for p in next_seg]
                next_y = [p[1] for p in next_seg]
                next_slope, next_intercept, r_value, p_value, std_err = linregress(next_x, next_y)
                if math.fabs(next_slope - slope0) > 0.5:
                    corners.append(next_seg[0])
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                next_x = [p[0] for p in next_seg]
                next_y = [p[1] for p in next_seg]
                next_slope, next_intercept, r_value, p_value, std_err = linregress(next_x, next_y)
                if math.fabs(next_slope - slope) > 0.5:
                    corners.append(next_seg[0])
            slope = next_slope
        # take left and right corner
        corner_l = (math.inf, 0)
        corner_r = (-math.inf, 0)
        for c in corners:
            if c[0] > corner_r[0]:
                corner_r = c
            if c[0] < corner_l[0]:
                corner_l = c
        # average y-values of corners
        c_avg = (corner_l[1] + corner_r[1]) / 2
        y = [p[1] for p in points]
        y_avg = (np.min(y) + np.max(y)) / 2
        # if (math.fabs(c_avg-y_avg)) <= 25:
        #     return 3
        # else:
        #     return 4
        return len(corners)

    def area_perimeter_ratio(self, area, perimeter, bbox):
        """
        Distinguish between basic shapes based on the ratio between the perimeter and area of the shape.
        :param area: area of the shape
        :param perimeter: perimeter of the shape
        :param bbox: bounding rectangle
        :return: (ratio, bool) where ratio is the value of the ratio that gives the smallest difference and bool denotes
        whether the ratio of the input is closer to the theoretical ratio of a rectangle (1) or closer to that of elipse (0), -1 in other cases
        """
        # Ratio of the input shape
        p_in = math.pow(perimeter, 2) / area
        w = bbox[1] - bbox[0]
        h = bbox[3] - bbox[2]
        s = w / h
        to_ret = (0, -1)
        min_val = math.inf
        p_d = 8 * (math.pow(s, 2) + 1) / math.pow(s, 2)  # For diamonds
        if math.fabs(p_in - p_d) < min_val:
            min_val = math.fabs(p_in - p_d)
            to_ret = p_d, -1
        p_r = 4 * (s + 2 + (1 / s))  # For rectangles
        if math.fabs(p_in - p_r) < min_val:
            min_val = math.fabs(p_in - p_r)
            to_ret = p_r, 0
        p_t = 2 * math.pow((s + math.sqrt(math.pow(s, 2) + 4)), 2)  # For triangles
        if math.fabs(p_in - p_t) < min_val:
            min_val = math.fabs(p_in - p_t)
            to_ret = p_t, -1
        K = 0.005095 * math.pow(s, 4) - 0.0693346 * math.pow(s, 3) + 0.346653 * s - 0.519223 * math.pow(s, 2) + 0.24308
        p_e = math.pi * ((s + 1) / 2) - K  # For circles
        if math.fabs(p_in - p_e) < min_val:
            min_val = math.fabs(p_in - p_e)
            to_ret = p_e, 1
        return to_ret

    def apply_filters(self, ar_per, ar_ac, s, r_e, p):
        """
        Apply the results of the filers.
        :param ar_per: area perimeter ratio
        :param ar_ac: ratio of the area defined by the convex hull and the bounding box
        :param s: segments from the gesture
        :param r_e: boolean from the area-perimeter filter that denotes the shape is a rectangle or ellipse
        :param p: the points in the gesture
        :return: void
        """
        # Order of filtering
        if 3 * math.pi <= ar_per <= 5 * math.pi:
            # Circle
            print("Circle")
        elif 35 <= ar_per < 55:
            # Undefined shape
            print("Unknown shape")
        elif 55 <= ar_per < 75:
            # Line
            print("Line")
        elif 0.35 <= ar_ac < 0.7:
            # Triangle/diamond
            ans = self.triangle_diamond_filter(s, p)
            if ans == 3:
                print("Triangle")
            elif ans == 4:
                print("Diamond")
            else:
                print("Unknown shape")
        elif 0.8 <= ar_ac <= 1 or r_e:
            # Rectangle
            print("Rectangle")
        else:
            # Elipse
            print("Elipse")

    def process_strokes(self):
        """
        Process the strokes from the gesture.
        :return: void
        """
        # Quick check to verify whether we are allowed to start processing.
        if self.update and (self.done or (
                self.moving and (datetime.now() - self.time).total_seconds() > self.timeout and not self.drawing)):
            self.update = False
            # Clear the points from previous calculations and add those from the current gesture
            points = []
            for list in self.strokes:
                for point in list:
                    points.append(point)
            # Save the gesture
            self.gestures.append((np.array(points), self.strokes))
            self.strokes = []
            # For each gesture stored, apply filters and perform classification
            for p, s in self.gestures:
                hull = ConvexHull(p, True)
                Ac = self.calc_area(p)
                x = p[hull.vertices, 0]
                y = p[hull.vertices, 1]
                bbox = np.min(x), np.max(x), np.min(y), np.max(y)
                # Calculate filters
                ar_ac = self.area_ratio_filter(x, y, Ac)
                area = self.calc_area(p)
                # area = hull.volume
                # perimeter = self.calc_perimeter(p)
                perimeter = hull.area
                ar_per, r_e = self.area_perimeter_ratio(area, perimeter, bbox)
                # Apply filters and draw the bounding box
                self.apply_filters(ar_per, ar_ac, s, r_e, p)
                self.canvas.create_rectangle(bbox[0], bbox[2], bbox[1], bbox[3])
            print("----------")


# Create window
root = Tk()
root.geometry("{0}x{1}".format(dim_x, dim_y))
recognizer = Recognizer()
# Run
root.mainloop()

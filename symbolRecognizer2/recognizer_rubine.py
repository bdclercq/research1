import tkinter
import numpy as np
import sc
import fv

dim_x = 950
dim_y = 750


class Recognizer(tkinter.Frame):
    def __init__(self):
        super().__init__()

        # Create canvas on window
        self.canvas = tkinter.Canvas(width=dim_x, height=dim_y)
        self.canvas.pack(expand=1)

        self.entry1 = tkinter.Entry(root)
        self.canvas.create_window(200, 140, window=self.entry1)

        self.label1 = tkinter.Label(root, text="Class name")
        self.canvas.create_window(200, 120, window=self.label1)

        self.button1 = tkinter.Button(text='Record example', command=self.training)
        self.canvas.create_window(200, 165, window=self.button1)

        self.init_canvas()

        self.classifier = sc.sClassifier()
        self.take_input = 0
        self.points = []
        self.ovals = []
        self.training_name = ""

    def init_canvas(self):
        # Bind actions
       # Release left mouse button
        self.canvas.bind('<ButtonRelease-1>', self.save_stroke)
        # Move the mouse while left button pressed
        self.canvas.bind('<B1-Motion>', self.stroke)
        self.canvas.pack()


    def training(self):
        name = self.entry1.get()
        if name == "" or name =="quit":
            self.classifier.sDoneAdding()
            self.classifier.write("classifier.out")
        else:
            self.take_input = 1
            self.training_name = name

    # Collects points while the left mouse button is pressed
    def stroke(self, event):
        if self.take_input:
            self.points.append((event.x, event.y, event.time))
            #python_green = "#476042"
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            oval = self.canvas.create_oval(x1, y1, x2, y2)
            self.ovals.append(oval)
            #print(self.points)

    def InputAGesture(self, gesture):
        feature_vector = fv.FV()
        for point in gesture:
            feature_vector.AddPoint(point[0], point[1], point[2])
        v = feature_vector.FvCalc()
        return v

    # Left mouse button is released, save stroke
    def save_stroke(self, event):
        #print("Saving stroke ", self.points)
        if self.take_input:
            self.take_input = 0
            self.classifier.sAddExample(self.training_name, self.InputAGesture(self.points))
            for o in self.ovals:
                self.canvas.delete(o)
            self.points = []
            self.ovals = []
            self.entry1.delete(0, 'end')


# Create window
root = tkinter.Tk()
root.geometry("{0}x{1}".format(dim_x, dim_y))

recognizer = Recognizer()

# Run
root.mainloop()

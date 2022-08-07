import tkinter

master=tkinter.Tk()
master.title("grid() method")
master.geometry("350x275")

button1=tkinter.Button(master, text="button1")
button1.grid(row=1,column=0)

button2=tkinter.Button(master, text="button2")
button2.grid(row=2,column=2)

button3=tkinter.Button(master, text="button3")
button3.grid(row=3,column=3)

button4=tkinter.Button(master, text="button4")
button4.grid(row=4,column=4)

button5=tkinter.Button(master, text="button5")
button5.grid(row=5,column=5)

master.mainloop()
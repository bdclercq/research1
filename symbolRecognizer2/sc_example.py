import sc
import fv_example

NEXAMPLES = 15


def take_input():
    pass


def MakeAClassifier():
    stat_clas = sc.sClassifier()
    while True:
        print("Enter class name, exit to exit: ")
        name = str(input())
        if name == "exit" or name == "":
            break
        for i in range(NEXAMPLES):
            print("Enter %s example %d\n", name, i)
            gesture = take_input()
            stat_clas.sAddExample(name, fv_example.InputAGesture(gesture))
    stat_clas.sDoneAdding()
    stat_clas.write("classifier.out")
    return stat_clas


def TestAClassifier(classifier):
    while True:
        pass

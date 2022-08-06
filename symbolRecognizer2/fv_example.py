import fv


def InputAGesture(gesture):
    feature_vector = fv.FV()
    for point in gesture:
        feature_vector.AddPoint(point[0], point[1], point[2])
    v = feature_vector.FvCalc()
    return v

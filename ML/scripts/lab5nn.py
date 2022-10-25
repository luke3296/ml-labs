import pandas as pd
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv(".\..\data\knn_example_data.csv", header=None)

# "ID","Class",Antennae Length","Abdomen Length"

random_state = 1
test_set_size = 0.3

# separate data into X, y structures to follow scikit-learn conventions
# X: features of the input data
# y: target/label/class vector of the input data
X = df[[2, 3]].values  # columns 2 and 3 are the instance attributes
y = df[1].values  # column 1 is the target/class


def euclidian_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    d1 = x2 - x1
    d2 = y2 - y1
    return math.sqrt(d1**2 + d2**2)

#where actual and predict are lists of 1's or 0's of the same length
def my_confusion_matrix(actual, predict):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(actual)):
        if (actual[i] == 1 and predict[i] == 1):
            tp += 1
        elif (actual[i] == 0 and predict[i] == 0):
            tn += 1
        elif (actual[i] == 0 and predict[i] == 1):
            fp += 1
        elif (actual[i] == 1 and predict[i] == 0):
            fn += 1
    return tp, fn, fp, tn


# X [2.7,5.5] list qusrt (2.7,5.5)
def predict(X, y, query):
    class_value = 0
    minDist = 100
    index = 0
    for instance in X:
        dist = euclidian_distance(instance, query)
        if dist < minDist:
            minDist = dist
            class_value = y[index]
        index += 1
    return class_value


#print(predict(X,y,(8.3,6.6)))

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=random_state)
print(X_train)
print(X_test)


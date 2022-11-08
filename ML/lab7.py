import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(filename):
    with open(filename,'rb') as f:
        training_data, validation_data, test_data = pickle.load(f,encoding='iso-8859-1')
        f.close()
        return (training_data, validation_data, test_data)
    
#takes a 1d 784 element ndarray as input and prints it as a 28 by 28 image
def print_img(arr):
    img1 = np.reshape(arr,(-1,28))
    plt.imshow(img1,aspect="auto")
    plt.show()


data = load_data("./mnist.pkl/mnist.pkl")
train_set, validation_set, test_set = data
print("num instances in train set: ",(len(train_set[1])));
print("num instances in validation set: ",(len(validation_set[1])));
print("num instances in test set: ",(len(test_set[1])));
print("the first image in the train set is\n" , train_set[0][0], " \n which represents a ",train_set[1][0])
print_img(train_set[0][0]);

train_set_imgs, train_set_lables = train_set
print((type(train_set_imgs)))
print((type(train_set_lables)))


import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from  sklearn.metrics import ConfusionMatrixDisplay , accuracy_score


def load_data(filename):
    with open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
        f.close()
        return (training_data, validation_data, test_data)


# takes a 1d 784 element ndarray as input and prints it as a 28 by 28 image
def print_img(arr):
    img1 = np.reshape(arr, (-1, 28))
    plt.imshow(img1, aspect="auto")
    plt.show()


data = load_data("./../data/mnist.pkl/mnist.pkl")
train_set, validation_set, test_set = data

train_set_imgs, train_set_lables = train_set
test_set_imgs, test_set_lables = test_set

random_seed = 1
#if  done with logistic and sgd the model predicts only 1's? when the model has more than 1 hidden layer
classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100,activation='logistic',solver='sgd',random_state=random_seed)
classifier.fit(train_set_imgs, train_set_lables)

pridictions = classifier.predict(test_set_imgs)
ascore = accuracy_score(test_set_lables,pridictions)
print("accuracy: ", ascore)

disp = ConfusionMatrixDisplay.from_predictions(test_set_lables,pridictions)

disp.figure_.suptitle("Confusion Matrix")
plt.savefig("cf_matrix.png")
plt.show()

activations =  ['identity', 'logistic', 'tanh', 'relu' ]
solver = ['lbfgs', 'sgd', 'adam']

for i in range(len(activations)):
    for j in range(len(solver)):
        classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, activation=activations[i], solver=solver[j],
                                   random_state=random_seed)
        classifier.fit(train_set_imgs, train_set_lables)
        pridictions = classifier.predict(test_set_imgs)
        ascore = accuracy_score(test_set_lables, pridictions)
        print("accuracy for 10 nurons with activation func : ",activations[i],  " and solver: ", solver[j], " is ", ascore )
        disp = ConfusionMatrixDisplay.from_predictions(test_set_lables, pridictions)

        disp.figure_.suptitle("Confusion Matrix " + activations[i] + " " +solver[j])
        plt.savefig("cf_matrix.png")
        plt.show()





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




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

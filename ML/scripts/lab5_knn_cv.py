import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # dataset
from sklearn.model_selection import train_test_split # split data
from sklearn.neighbors import KNeighborsClassifier # knn classifier
from sklearn.model_selection import cross_val_score # cross validation
from sklearn.metrics import accuracy_score # compute accuracy score
# load ALL data, where
# X: features of the input data
# y: target/label/class vector of the input data
X, y = load_breast_cancer(return_X_y=True)
# split data into train_validation and test sets
train_validation_random_seed = 1
test_set_size = 0.3
X_train_validation, X_test, y_train_validation, y_test = \
train_test_split(X, y, test_size=test_set_size,
random_state=train_validation_random_seed)

kfold_splits = 10 # number of folds

# generate a list of k values to test
number_k_models = 100 # number of k-Nearest Neighbours
# generate data structures to loop through the k values and
# store the corresponding accuracy scores
k_neighbours = [i for i in range(1, number_k_models+1)]
k_neighbours_accuracy_scores = []
# init best_score variable
best_score = 0
# loop through the k_neighbours, perform cross-validation and compute the accuracy
for k in k_neighbours:
    # create a KNN classifier instance
    knn = KNeighborsClassifier(n_neighbors=k)
    # perform cross-validation
    scores = cross_val_score(knn, X_train_validation, y_train_validation,
    cv=kfold_splits, scoring='accuracy', verbose=True)
    # compute the mean cross-validation accuracy.
    # there are 10 accuracy scores here as the number of folds!
    k_accuracy_score = np.mean(scores)
    # save the accuracy for the actual k
    k_neighbours_accuracy_scores.append(k_accuracy_score)
    # update the best score across the k_neighbours
    # if got a higher accuracy, store the accuracy score and the value of k
    if k_accuracy_score > best_score:
        best_score = k_accuracy_score
        best_k = k
# optimisation process with cross validation on train_validation set ends here
# print the best accuracy and the best k
print(f"Highest accuracy: {best_score} when k={best_k}")

# requires to implement a 'training', which actually load your train_validation data
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_validation, y_train_validation) # implement the 'fitting' to load the combined train_val
# predict values on the test set
y_predictions = knn.predict(X_test)
knn_accuracy_score = accuracy_score(y_test, y_predictions)
print(f"Accuracy on test set: {knn_accuracy_score}")

# plot accuracy scores
plt.xlim(1, number_k_models)
plt.ylim(0.7, 1.0)
plt.ylabel('Accuracy score')
plt.xlabel('k')
plt.plot(np.arange(1, number_k_models+1, 1), k_neighbours_accuracy_scores, linestyle='-', color='red')
plt.show()
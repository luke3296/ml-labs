from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

random_seed = 1

X, y = load_breast_cancer(return_X_y=True);

print('Number of instances', X.shape[0])
print('Numer of attributes', X.shape[1])

split_function = ShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
instance_indexed_list = list(split_function.split(X, y))
train_index_set = instance_indexed_list[0][0]
test_index_set = instance_indexed_list[0][1]

X_train = X[train_index_set, :]
y_train = y[train_index_set]
X_test  = X[test_index_set, :]
y_test = y[test_index_set]

print(' Training set size ', train_index_set.shape[0])
print(' Test set size', test_index_set.shape[0]) # this gives a different number to the lab sheet

number_trees = 500;
dt_training_samples = 200;
bagging_ens = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=number_trees,
    max_samples=dt_training_samples,
    bootstrap=True,
    random_state=random_seed)

bagging_ens.fit(X_train, y_train)
y_predictions = bagging_ens.predict(X_test);
accurarcy = accuracy_score(y_test, y_predictions);

print('Accuracy (testing set vs predictions):', accurarcy)

bagging_ens = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=number_trees,
    max_samples=dt_training_samples,
    bootstrap=True,
    oob_score=True,
    random_state=random_seed)

bagging_ens.fit(X_train, y_train)

print('Out-Of-Bag (OOB) score', bagging_ens.oob_score_)

rf_classifier = RandomForestClassifier(
    random_state=random_seed)

rf_classifier.fit(X_train, y_train)
y_preidictions_rf = rf_classifier.predict(X_test);
accuracy_rf = accuracy_score(y_test, y_preidictions_rf)

print('Accuracy (Random Forest):', accuracy_rf)


number_trees = 200
rf_classifier = RandomForestClassifier(
n_estimators=number_trees,
random_state=random_seed
)
rf_classifier.fit(X_train, y_train)
# accuracy on test set
y_predictions_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_predictions_rf)
print('Accuracy (Random Forest):', accuracy_rf)


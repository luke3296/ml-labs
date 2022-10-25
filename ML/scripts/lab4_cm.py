from sklearn.datasets import load_breast_cancer
import sklearn.metrics
import pandas as pd

predicted_df = pd.read_csv(".\..\data\Week4\predictions.csv", header=None, index_col=False, sep=',');
actual_df = pd.read_csv(".\..\data\Week4\\test_target.csv", header=None, index_col=False, sep=',');

pridicted = predicted_df[0].values
actual = actual_df[0].values

X, y = load_breast_cancer(return_X_y=True);

def my_confusion_matrix(actual, predict):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(actual)):
        if (actual[i] == 1 and predict[i] == 1):
                tp+=1
        elif (actual[i] == 0 and predict[i] == 0):
                tn+=1
        elif (actual[i] == 0 and  predict[i] == 1):
                fp+=1
        elif (actual[i] == 1 and  predict[i] == 0):
                fn+=1
    return tp, fn, fp, tn

(tp, fn, fp, tn) = my_confusion_matrix(actual, pridicted)

print(tp)
print(fn)
print(fp)
print(tn)

TPR = tp / (tp+fn)
TNR = tn / (fp+tn)
Accuracy = (tp + tn) / (tp+tn+fp+fn)
Balanced_Accuracy = (TPR + TNR) / 2
Precision = tp / (tp + fp)
Sensitivity = tp / (tp + fn)
Specificity = tn / (fp + tn)


print("TPR: ",  TPR)
print("TNR: ",  TNR)
print("Accuracy: ",  Accuracy)
print("Balanced_Accuracy: ",  Balanced_Accuracy)
print("Precision: ",  Precision)
print("Sensitivity: ",  Sensitivity)
print("Specificity: ",  Specificity)

print('################')



# load packages
# data manipulation and plot
import pandas as pd
import matplotlib.pyplot as plt
# confusion-matrix functions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
# # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# # read a csv file and return the first column
def read_csv(filename):
    df = pd.read_csv(filename, header=None, index_col=False, sep=',')
    return df[0].values
# read data, actual values and predictions
y_test = read_csv('.\..\data\Week4\\test_target.csv')
y_predict = read_csv('.\..\data\Week4\predictions.csv')


# # compute confusion matrix and get the counts
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
specificity = tn / (tn+fp)
print('TP', tp)
print('FN', fn)
print('FP', fp)
print('TN', tn)
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_predict))
print('Specificity:', specificity)
print('Report\n', classification_report(y_test, y_predict))
# visualise the confusion matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
print('cm\n', cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot()
plt.show()



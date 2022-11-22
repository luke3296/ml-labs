#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:27:37 2022

@author: luke
"""
"""
STUDENT_ID: 100193308
Created on: Sun Nov  6 20:54:27 2022
Last update: Sun Nov  6 2022, created this comment block
Description: implments the cw
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("./../data/breast-cancer.data", header=None)
#print(data)
#preprocess the data
#missing values denated with ?
print(len(data))
print(data.head(10))
# data[atternum][instance_num]  data[col][row]
data = data.dropna()   
dropValues=[]
newIndex=[]
count=0
for i in range(len(data)):
    if (data[0][i]=='?' or data[1][i]=='?' or data[2][i]=='?' or data[3][i]=='?' or data[4][i]=='?' or data[5][i]=='?' or data[6][i]=='?' or data[7][i]=='?' or data[8][i]=='?' or data[9][i]=='?'):
        dropValues.append(i)
    else:
        newIndex.append(count)
        count+=1
        #data.drop(index=[i], axis=0, inplace=True)
        #data.reset_index()
        
df3=data.drop(dropValues)
df3 = df3.dropna()   
print(len(data))
print(len(df3))
#df3 has undefined values removed
df3 = df3.reindex(newIndex)
df3 = df3.dropna()   


#df3 = df3.astype({0:'int'})
def convertCol(str_in):
    if str_in == 'no-recurrence-events':
        return 0
    elif str_in == 'recurrence-events':
        return 1
    elif str_in == 'no':
        return 0
    elif str_in == 'yes':
        return 1
    elif str_in == 'lt40':
        return 0
    elif str_in == 'ge40':
        return 1
    elif str_in == 'premeno':
        return 2
    elif str_in == 'left':
        return 0
    elif str_in == 'right':
        return 1
    elif str_in == 'left_up':
        return 0
    elif str_in == 'left_low':
        return 1
    elif str_in == 'right_up':
        return 2
    elif str_in == 'right_low':
        return 3
    elif str_in == 'central':
        return 4
    
ageGroups=['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
tumourSizes=['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44','45-49', '50-54', '55-59']
inv_nodes=['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26','27-29', '30-32', '33-35', '36-39']

def convertColAge(str_in):
    count=0
    arr=ageGroups
    for str_test in arr:
        if str_test==str_in:
            return count
        count+=1    
        
def convertColTsize(str_in):
    count=0
    arr=tumourSizes
    for str_test in arr:
        if str_test==str_in:
            return count
        count+=1    
    
def convertColInvNodes(str_in):
    count=0
    arr=inv_nodes
    for str_test in arr:
        if str_test==str_in:
            return count
        count+=1 
        
#y is a list length of clean data, with 0 for no-recurrence-events and 1 for recurrence-events
y = list(map(convertCol , df3[0]))
df3[2] = df3[2].map(convertCol)
df3[5] = df3[5].map(convertCol)
df3[7] = df3[7].map(convertCol)
df3[8] = df3[8].map(convertCol)
df3[9] = df3[9].map(convertCol)

df3[1] = df3[1].map(convertColAge)
df3[3] = df3[3].map(convertColTsize)
df3[4] = df3[4].map(convertColInvNodes)

X = df3.iloc[:, [1,2,3,4,5,6,7,8,9]]

randomSeed=0
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.3, random_state=randomSeed)
print(X_train)
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
bal_acc_unseen = balanced_accuracy_score(y_test, y_pred)
y_pred_seen=classifier.predict(X_train)
bal_acc_seen = balanced_accuracy_score(y_train, y_pred_seen)
print(f'dt_balanced_acc_{bal_acc_seen}/{bal_acc_unseen}')
with open('./../output/dt_balanced_acc.txt', 'a') as dt_file:
    dt_file.write(f'dt_balanced_acc_{bal_acc_seen}/{bal_acc_unseen}')


MaxTreeDepth = 11;
acc=[]

with open('./../output/dt_balanced_acc_scores.txt', 'a') as tree_bal:
    for i in range(1,MaxTreeDepth):
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        classifier.fit(X_train,y_train)
        y_pred=classifier.predict(X_test)
        bal_acc_unseen = balanced_accuracy_score(y_test, y_pred)
        acc.append(bal_acc_unseen)
        tree_bal.write(f'dt_entropy_max_depth_{i}_balanced_acc = {bal_acc_unseen}\n')
        print(f'dt_entropy_max_depth_{i}_balanced_acc = {bal_acc_unseen}')
print(len(acc))
        
treeNum=[1,2,3,4,5,6,7,8,9,10]
plt.scatter(treeNum, acc)
plt.savefig("./../output/dt_balanced_acc_scores.png")

# best accuracy was seen with max_depth =4 

classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier, filled=True)
plt.savefig("./../output/dt_entropy_max_depth_4.png")

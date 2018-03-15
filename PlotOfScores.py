#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:08:34 2017

@author: abhilasha

The following code enables you to use various classifiers where features  and labels are used to train the classifiers.
"""



import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

#feature_file = csv.reader(open('features.txt'), delimiter=" ")
label_file = csv.reader(open('labels.txt'),delimiter = " ")
data_y = []
score_list=[]
objects = ('NuSVC', 'SVM.svc', 'Logistic','Gradient','Decision','AdaBoost','Linear')
y_pos = np.arange(len(objects))
data_X = np.loadtxt('features.txt')
print (data_X, data_X.shape)

for row in label_file:
	data_y.append(row)

data_y = np.array(data_y).astype(np.float)
print (data_y.shape,data_y)

train_X1 = data_X[0:50,:]
train_X2 = data_X[81:,:]
train_y1 = data_y[0:50,:]
train_y2 = data_y[81:,:]
X_train = np.concatenate((train_X1,train_X2),axis=0)
y_train = np.concatenate((train_y1,train_y2),axis=0)
y_train = y_train.reshape(-1)

X_test = data_X[50:80,:]
y_test = data_y[50:80,:]
y_test = y_test.reshape(-1)
#print (X_train.shape,y_train.shape)
#print (X_test.shape,y_test.shape)

clf = svm.NuSVC(nu=0.2)
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = svm.SVC(C=1, kernel='sigmoid', degree=2, gamma=100)
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = LogisticRegression(C=1, penalty='l1')
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = GradientBoostingClassifier()
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = AdaBoostClassifier(n_estimators = 100)
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
clf = svm.LinearSVC(C=100000)
clf.fit(X_train,y_train)
score_list.append(clf.score(X_test,y_test))
print(score_list)

plt.bar(y_pos, score_list, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Value of Classifier Scores')
plt.title('Classifier Comparision Plot')
plt.show()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import gen_data
df_com, df_sin, df_hosp = gen_data.gen_data()

df_com.head()
df_sin.head()
df_hosp.head()

def get_conf(X, y, classifier):
    conf = [[0,0],[0,0]]
    for clf in [0, 1]:
        for pred_clf in [0, 1]:
            X_clf = X[y == clf]
            y_clf = y[y == clf]
            y_pred = classifier.predict(X_clf)
            cnt = len(y_pred[y_pred == pred_clf])
            prob = cnt/len(y_clf)
            conf[clf][pred_clf] = prob
    TP, FP, FN, TN = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    fm = 2*TP/(2*TP+FP+FN)
    recall = TP/(TP+FN)
    print("Accuaracy: ", acc)
    print("Precision: ", prec)
    print("f-measure: ", fm)
    print("Recall: ", recall)
    return conf 

def predict(X, y, gen_clf):
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
    X_split = np.concatenate((X[y==0][:len(y[y==1])], X[y==1]))
    y_split = np.concatenate((y[y==0][:len(y[y==1])], y[y==1]))
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2)
    clf = gen_clf()
    y_pred = clf.fit(X_train, y_train)
    
    print(clf.score(X_val, y_val))
    conf = get_conf(X_val, y_val, clf)
    for row in conf:
        print(row)


for df in (df_com, df_sin, df_hosp):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    predict(X, y, GaussianNB)
    print()

X, y = df_hosp.iloc[:, :-1], df_hosp.iloc[:,-1]
X_split = np.concatenate((X[y==0][:len(y[y==1])], X[y==1]))
y_split = np.concatenate((y[y==0][:len(y[y==1])], y[y==1]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.3, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)

gnb.score(X_train, y_train)

def get_conf(X_test, y_test, clf):
    conf = [[0,0],[0,0]]
    for clfs in [0, 1]:
        for pred_clf in [0, 1]:
            X_test_clf = X_test[y_test == clfs]
            y_test_clf = y_test[y_test == clfs]
            y_pred =  clf.predict(X_test_clf)
            cnt = len(y_pred[y_pred == pred_clf])
            prob = cnt/len(y_test_clf)
            conf[clfs][pred_clf] = prob
    return conf 

print(gnb.score(X_test, y_test))
conf = get_conf(X_test, y_test, gnb)
for row in conf:
    print(row)

print(gnb.score(X, y))
conf = get_conf(X, y, gnb)
for row in conf:
    print(row)




# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:00:47 2021

@author: Jb
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn

def print_report(labels, true, predicted):
    class_report = classification_report(true, predicted, target_names=labels)
    conf_matrix = confusion_matrix(true, predicted)
    print('\nClass Report: \n', class_report)
    plt.figure(2)
    ax = plt.subplot()
    df_cm = pd.DataFrame(conf_matrix, index=labels,columns=labels)
    # sn.set(font_scale=1.0)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()
    
x = np.load('modified_x.npy')
y = np.load('modified_y.npy')
verbose = True
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

clf = LinearDiscriminantAnalysis()
k_folds = 4
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

avg_list = []
labels = ['NH3', 'NO2', 'Mixture1', 'Mixture2']
for i in range(10):
    true = []
    predicted = []
    scores = []
    for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)): 
        x_train = x[train_ids]
        x_test = x[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]
    
        clf.fit(x_train, y_train)
        if (verbose):
            print('Fold ', fold, ': ', clf.score(x_test, y_test)*100, '%\n', end='')
    
        scores.append(clf.score(x_test, y_test))
        true += list(y_test)
        predicted += list(clf.predict(x_test))
    fold_avg = np.average(scores)*100
    avg_list.append(fold_avg)
    if (verbose):
        print("Average of the folds:", fold_avg)
        print_report(labels, true, predicted)
print("Avg: %.2f +- %.2f" % (np.average(avg_list), np.std(avg_list)))

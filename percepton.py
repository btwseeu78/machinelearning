# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:41:20 2016

@author: Apu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class Percepton(object):
    def __init__(self, learning_rate, nof_iter):
        self.learning_rate = learning_rate
        self.nof_iter = nof_iter

    def fit(self, X, y):
        """ parametrs
        X = [n_samples,n_features]
        is training vector,y= the outcome vector [n_samples]
        """
        self.weight_after_fitting = np.zeros(1+X.shape[1])
        
        self.errors_ = []
        for _ in range(self.nof_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate*(target - self.predict(xi))
                
                
                self.weight_after_fitting[1:] += update * xi
                self.weight_after_fitting[0] += update
                
                errors += int(update != 0.0)
                # print("target:", target, "xi:", xi, "update:", update, "errors:", errors, "prdecit:", self.predict(xi), "\t\t weight:", self.weight_after_fitting)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        m = np.dot(X, self.weight_after_fitting[1:]) + self.weight_after_fitting[0]
        return m

    def predict(self,X):
        """ return class label after unit step function
        """
        q = self.net_input(X)
       
        return np.where(self.net_input(X) >= 0.0, 1, -1)
            

def plot_dec(X, y,classifier, resolution=0.02):
    markers = (i for i in "sxo^v")
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    
    

            
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y=np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker='x', label= 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal lengtg')
plt.legend(loc='upper left')
plt.show()
ppn=Percepton(learning_rate=0.1,nof_iter=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_, marker = 'o')
plt.xlabel('Epocha')
plt.ylabel('no of mislassifiction')
plt.show()
m=ppn.errors_



    

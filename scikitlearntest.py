# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:02:13 2017

@author: thomas
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.externals.six import StringIO

from sklearn import tree

X = [[i] for i in range(100)]
print(X)

y = [[i] for i in range(100)]
print(y)

X, y = make_regression(n_features=1, n_informative=2,random_state=0, shuffle=False)

print(X)
print(y)

regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=50, min_samples_split=100, max_leaf_nodes=100)

regr.fit(X, y)

'''
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
'''
print(regr.predict([[40]]))




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:19:14 2020
@brief Music Recommendation ML project
@author: swapnil
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1.Import the data
music_data = pd.read_csv('music.csv')
# 2. Clean the data
in_set = music_data.drop(columns=['genre'])
out_set=music_data['genre']
# 3. split the data
in_train, in_test, out_train, out_test = train_test_split(in_set, out_set, test_size=0.2)
# 4. create a model
model = DecisionTreeClassifier()
# 5. train the model
model.fit(in_train, out_train)
# 6. Make a Predictions
predictions=model.predict(in_test)
# 7. Evaluation
sc = accuracy_score(out_test, predictions)
print(sc)

''' 
#without splitting the data
model.fit(in_set, out_set)
predictions = model.predict([[21,1],[22,0]])
'''
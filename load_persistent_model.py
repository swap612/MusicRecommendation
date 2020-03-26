#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:40:42 2020
@brief: Loading persistent trained model and predicting
@author: swapnil
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# 1.Import the data
music_data = pd.read_csv('music.csv')
# 2. Clean the data
in_set = music_data.drop(columns=['genre'])
out_set=music_data['genre']
# 3. split the data
in_train, in_test, out_train, out_test = train_test_split(in_set, out_set, test_size=0.2)
# 4. load the model
model = joblib.load('music-recommender.joblib')
# 5. Make a Predictions
predictions=model.predict(in_test)
# 6. Evaluation
sc = accuracy_score(out_test, predictions)
print(sc)
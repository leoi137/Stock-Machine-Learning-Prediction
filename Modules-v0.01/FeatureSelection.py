import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

class Feature_and_Splitting():

    """
    Parameters:

    X: Features, the values used to learn from

    y: The value being predicted
    
    test_size: Amount of data wanted to not be trained on 
    and kept for testing it is a value between 0 to 1.

    K_feat: The number of features wants

    columns: Original columns before feature selection

    shuffle: Shuffle data during train_test_split
    """

    def __init__(self, X, y, test_size, K_feat, columns, shuffle = False):

        self.X = X
        self.y = y
        self.test_size = test_size
        self.K_feat = K_feat
        self.shuffle = shuffle
        self.columns = columns
        
    def main(self, kind = 'Classification'):

        print("Feature Selection...")
        print("-------------------------")
        start_all = time.perf_counter()

        X_train, X_test, y_train, y_test = self.train_split()

        try:
            new_features, dropped, after_feat_sc, X_train_selected, X_test_selected, y_train, y_test = self.feature_selection(
                X_train, X_test, y_train, y_test, kind = kind)
        except ValueError:
            new_features, dropped, after_feat_sc, y_scaler, X_train_selected, X_test_selected, y_train, y_test = self.feature_selection(
                X_train, X_test, y_train, y_test, kind = kind)

        print("Feature Selection took: {:0.4f} seconds\n".format(time.perf_counter() - start_all))

        if kind == 'Classification':   
            return new_features, dropped, after_feat_sc, X_train_selected, X_test_selected, y_train, y_test
        else:
            return new_features, dropped, after_feat_sc, y_scaler, X_train_selected, X_test_selected, y_train, y_test

    def train_split(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 0, shuffle = self.shuffle)

        return X_train, X_test, y_train, y_test


    def feature_selection(self, X_train, X_test, y_train, y_test,  kind, fit_on = 'Test'):

        y_scaler = StandardScaler()

        if kind == 'Regression':
            y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
            y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

        feat_scaler = StandardScaler()

        if fit_on == 'Test':
            self.features = X_test
            self.target = y_test

        else:
            self.features = X_train
            self.target = y_train

        self.X_vals = feat_scaler.fit_transform(self.features)

        self.select = SelectKBest(k = self.K_feat)
        self.select.fit(self.X_vals, self.target)

        mask = self.select.get_support()

        new_features = []
        dropped = []
        
        for bool, feature in zip(mask, self.columns):
            if bool:
                new_features.append(feature)
            else:
                dropped.append(feature)

        after_feat_sc = StandardScaler()

        X_train_selected = X_train.drop(dropped, axis = 1)
        X_test_selected = X_test.drop(dropped, axis = 1)

        X_train_selected = after_feat_sc.fit_transform(X_train_selected)
        X_test_selected = after_feat_sc.transform(X_test_selected)

        if kind == 'Regression':
            return new_features, dropped, after_feat_sc, y_scaler, X_train_selected, X_test_selected, y_train, y_test

        else:
            return new_features, dropped, after_feat_sc, X_train_selected, X_test_selected, y_train, y_test
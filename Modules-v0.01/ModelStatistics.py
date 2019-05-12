import pandas as pd
import numpy as np
import json
import pickle
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Statistics:

    """
    Parameters:
    
    THIS ONLY WORKS WITH A CLASSIFIER MODEL WITH 3 PREDICTED VALUES

    X_test_selected: The selected featues to test on (called selected because it assumes
    they have gone through a feature selection process, works either way.)

    y_test: The value being predicted for testing

    Model: The trained model

    model_features: The name of the features used

    X_scaler: Scaler used for the X values

    y_scaler: Scaler used for y values (typically only for regression)

    scores: The results after running k-Fold cross validation

    model_dir: The directory of where you want to save trained models

    size: Only supports 3 and 2, meaning three or two different predicted values

    """
    def __init__(self, X_test_selected, y_test, Model, model_features, X_scaler, y_scaler, scores, model_dir = os.getcwd(), size = 3):

        self.X_test_selected = X_test_selected
        self.y_test = y_test
        self.size = size
        self.Model = Model
        self.model_feat = model_features
        self.scores = scores
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.model_dir = model_dir
        
        self.data_dict = {}
        
    def save_all(self, model_name, nick_name):

        self.create_dir(nick_name)
        self.save_scalers(model_name, nick_name)
        self.save_file(model_name, nick_name)
        self.save_model(model_name, nick_name)

    def statistics(self):
        y_preds = self.Model.predict(self.X_test_selected)
        conf_matrix = confusion_matrix(self.y_test, y_preds)
        # conf_DF = pd.DataFrame(conf_matrix, columns = ['-1', '0', '1'], index = ['-1', '0', '1'])
        if self.size == 3:
            bear_mean = (conf_matrix[0, 0]/ (conf_matrix[0, 0] + conf_matrix[1, 0] + conf_matrix[2, 0]))
            bull_mean = (conf_matrix[2, 2]/ (conf_matrix[0, 2] + conf_matrix[1, 2] + conf_matrix[2, 2]))
            none_mean = (conf_matrix[1, 1]/ (conf_matrix[0, 1] + conf_matrix[1, 1] + conf_matrix[2, 1]))
            
            self.data_dict['Accuracy'] = {'All': round(accuracy_score(y_preds, self.y_test), 4), 
                                          'Bull': round(bull_mean, 4), 
                                          'Bear': round(bear_mean, 4), 
                                          'Stalled': round(none_mean, 4), 
                                          'STDV': self.scores.std()}
            self.data_dict['Confusion Matrix'] = {'-1': (int(conf_matrix[0, 0]), int(conf_matrix[1, 0]), 
                                                         int(conf_matrix[2, 0])), 
                                                 '0': (int(conf_matrix[0, 1]), int(conf_matrix[1, 1]), 
                                                       int(conf_matrix[2, 1])), 
                                                 '1': (int(conf_matrix[0, 2]), int(conf_matrix[1, 2]), 
                                                       int(conf_matrix[2, 2]))}
            self.data_dict['Features'] = self.model_feat

            self.conf_m = pd.DataFrame(self.data_dict['Confusion Matrix'])
            bear = np.array(self.conf_m['-1']).sum()
            bull = np.array(self.conf_m['1']).sum()
            total_data = len(self.X_test_selected)
            print(f"Total Predictions: {(bear + bull)/total_data:0.4f}%")     

        if self.size == 2:
            y_preds = self.Model.predict(self.X_test_selected)
            conf_matrix = confusion_matrix(self.y_test, y_preds)

            pred_acc = (conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1]))
            none_acc = (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]))

            self.data_dict['Accuracy'] = {'All': round(accuracy_score(y_preds, self.y_test), 4), 
                                          'Bull': round(pred_acc, 4), 
                                          'Stalled': round(none_acc, 4), 
                                          'STDV': self.scores.std()}

            self.data_dict['Confusion Matrix'] = {'0': (int(conf_matrix[0, 0]), int(conf_matrix[1, 0])), 
                                                 '1': (int(conf_matrix[0, 1]), int(conf_matrix[1, 1]))}

            self.data_dict['Features'] = self.model_feat

            self.conf_m = pd.DataFrame(self.data_dict['Confusion Matrix'])
            preds = np.array(self.conf_m['1']).sum()
            total_data = len(self.X_test_selected)
            print(f"Total Predictions: {preds/total_data:0.4f}%") 
            
        return self.data_dict

    def create_dir(self, nick_name):

        path = f"{self.model_dir}/{nick_name}"

        try:  
            os.mkdir(path)
        except FileExistsError:
            pass
        except OSError:  
            print(f"Creation of the directory {path} failed")
        else:  
            print(f"Successfully created the directory {path}")

    def save_scalers(self, model_name, nick_name):

        scalers = {}
        scalers['Scalers'] = {"X_scaler": self.X_scaler, "y_scaler": self.y_scaler}
        pickle.dump(scalers, open(f"{self.model_dir}/{nick_name}/SCALERS{model_name}.pickle", 'wb'))
        
    def save_file(self, model_name, nick_name):
        """
        Parameters:

        model_name: The name of a file to save it as in string format
        """
        stats = self.statistics()
        with open(f"{self.model_dir}/{nick_name}/{model_name}.json", 'w') as outfile:
            json.dump(stats, outfile)

    def save_model(self, model_name, nick_name):
        pickle.dump(self.Model, open(f"{self.model_dir}/{nick_name}/{model_name}.pickle", 'wb'))

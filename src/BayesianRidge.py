import yaml
import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.linear_model import BayesianRidge
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Input, Dense, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import preprocessing
import dataloader
from preprocessing import preprocessor
from dataloader import dataLoader
import sys
import os
import os.path
import csv
from sklearn.neighbors import KNeighborsRegressor
#import ensemRegressor
#import optunatransformator1
import util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
class BayesianRidge():
  def __init__(self, num_of_estimators):
    self.num_of_estimators = int(num_of_estimators)
  def __call__(self, config_yaml, A_X_train, A_Y_train, A_tar_x_scaled, A_tar_y_scaled, B_X_train, B_Y_train, B_tar_x_scaled, B_tar_y_scaled):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    #callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=40)
    folder_name = "RF"
    #tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=i*50)
    """
    """
    #clf1 = RandomForestRegressor(n_estimators=self.num_of_estimators)
    #clf1.fit(source_features, source_labels.ravel())
    parameters = { 'max_iter': [50, 100, 500, 1000],
               'num_of_estimators': [1, 5, 10],
               'alpha_1': [0.00001, 0.0001, 0.001, 0.01, 0.1],
               'alpha_2': [0.00001, 0.0001, 0.001, 0.01, 0.1],
               'lambda_1': [0.00001, 0.0001, 0.001, 0.01, 0.1],
               'lambda_2': [0.00001, 0.0001, 0.001, 0.01, 0.1]} 
    grid = GridSearchCV(BayesianRidge(),parameters)
    model = grid.fit(A_X_train, A_Y_train)
    print(model.best_params_,'\n')
    print(model.best_estimator_,'\n')    

    params = model.best_params_

    rp_source_model = BayesianRidge(**model.best_params_)
    rp_source_model.fit(A_X_train, A_Y_train)
    yhat2 = rp_source_model.predict(A_tar_x_scaled)
    mse3 = mean_squared_error(A_tar_y_scaled, yhat2)
    mape3 = mean_absolute_percentage_error(A_tar_y_scaled, yhat2)
    print(mse3)
    print(mape3)
    
    grid = GridSearchCV(BayesianRidge(),parameters)
    model = grid.fit(B_X_train, B_Y_train)
    print(model.best_params_,'\n')
    print(model.best_estimator_,'\n')



    rc_source_model = BayesianRidge(**params)
    rc_source_model.fit(B_X_train, B_Y_train)
    yhat4 = rc_source_model.predict(B_tar_x_scaled)
    mse5 = mean_squared_error(B_tar_y_scaled, yhat4)
    mape5 = mean_absolute_percentage_error(B_tar_y_scaled, yhat4)
    return mse3, mape3, mse5, mape5







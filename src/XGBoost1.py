import yaml
import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.ensemble import GradientBoostingRegressor
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
from sklearn.impute import SimpleImputer

class XGBoost():
  def __init__(self, num_of_estimators):
    self.num_of_estimators = int(num_of_estimators)
  def __call__(self, config_yaml, A_X_train, A_Y_train, A_tar_x_scaled, A_tar_y_scaled, B_X_train, B_Y_train, B_tar_x_scaled, B_tar_y_scaled):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    #callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=40)
    folder_name = "RF"
    imputer_A = SimpleImputer(strategy='mean')
    A_X_train = imputer_A.fit_transform(A_X_train)
    A_tar_x_scaled = imputer_A.transform(A_tar_x_scaled)

    B_X_train_df = pd.DataFrame(B_X_train)
    B_tar_x_scaled_df = pd.DataFrame(B_tar_x_scaled)
    B_X_train_df, B_tar_x_scaled_df = B_X_train_df.align(B_tar_x_scaled_df, join='outer', axis=1)
    B_X_train_df = B_X_train_df.fillna(np.nan)
    B_tar_x_scaled_df = B_tar_x_scaled_df.fillna(np.nan)

    B_X_train = B_X_train_df.values
    B_tar_x_scaled = B_tar_x_scaled_df.values
    B_combined = np.vstack((B_X_train, B_tar_x_scaled))
    imputer_B = SimpleImputer(strategy='mean')
    B_combined_imputed = imputer_B.fit_transform(B_combined)
    B_X_train = B_combined_imputed[:B_X_train.shape[0], :]
    B_tar_x_scaled = B_combined_imputed[B_X_train.shape[0]:, :]
    #tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=i*50)
    """
    """
    #clf1 = RandomForestRegressor(n_estimators=self.num_of_estimators)
    #clf1.fit(source_features, source_labels.ravel())
    def objective(trial):
    # Define the hyperparameters for GradientBoostingRegressor
      params = { 
          'n_estimators': trial.suggest_int('n_estimators', 50, 200),
          'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
          'max_depth': trial.suggest_int('max_depth', 3, 15),
          'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
          'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
          'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
          'max_features': trial.suggest_uniform('max_features', 0.5, 1.0),
          'alpha': trial.suggest_uniform('alpha', 0.75, 0.99)
      }

      # Train model on dataset A
      rp_source_model = GradientBoostingRegressor(**params)
      rp_source_model.fit(A_X_train, A_Y_train)
      yhat2 = rp_source_model.predict(A_tar_x_scaled)
      mse3 = mean_squared_error(A_tar_y_scaled, yhat2)
      mape3 = mean_absolute_percentage_error(A_tar_y_scaled, yhat2)

      # Train model on dataset B
      rc_source_model = GradientBoostingRegressor(**params)
      rc_source_model.fit(B_X_train, B_Y_train)
      yhat4 = rc_source_model.predict(B_tar_x_scaled)
      mse5 = mean_squared_error(B_tar_y_scaled, yhat4)
      mape5 = mean_absolute_percentage_error(B_tar_y_scaled, yhat4)

      # Return the combined error (mse3 + mse5) for Optuna to minimize
      return mse3 + mse5

    # Create the Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Get the best hyperparameters from the study
    best_params = study.best_params

    # Train final models with the best hyperparameters
    rp_source_model = GradientBoostingRegressor(**best_params)
    rp_source_model.fit(A_X_train, A_Y_train)
    yhat2 = rp_source_model.predict(A_tar_x_scaled)
    mse3 = mean_squared_error(A_tar_y_scaled, yhat2)
    mape3 = mean_absolute_percentage_error(A_tar_y_scaled, yhat2)

    rc_source_model = GradientBoostingRegressor(**best_params)
    rc_source_model.fit(B_X_train, B_Y_train)
    yhat4 = rc_source_model.predict(B_tar_x_scaled)
    mse5 = mean_squared_error(B_tar_y_scaled, yhat4)
    mape5 = mean_absolute_percentage_error(B_tar_y_scaled, yhat4)

    # Output the best parameters and performance metrics
    print(f"Best parameters obtained from Optuna: {best_params}")
    print(f"MSE for dataset A: {mse3}, MAPE for dataset A: {mape3}")
    print(f"MSE for dataset B: {mse5}, MAPE for dataset B: {mape5}")

    # Return the final evaluation metrics
    return mse3, mape3, mse5, mape5
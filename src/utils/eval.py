import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import xgboost as xgb 
from sklearn.metrics import mean_squared_error

def evaluate(xgb_model: object, x_train, x_test, y_train, y_test):
    # xgboost models 
    dtrain = xgb.DMatrix(x_train)
    pred_tr = xgb_model.predict(dtrain, ntree_limit=xgb_model.best_ntree_limit)
    dtest = xgb.DMatrix(x_test)
    pred = xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit)
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title("train")
    plt.scatter(np.arange(len(pred_tr)).tolist(), pred_tr-y_train.values.ravel())
    plt.plot(np.arange(len(pred_tr)).tolist(), np.zeros(len(pred_tr)), color="r")
    plt.subplot(1, 2, 2)
    plt.title("test")
    plt.scatter(np.arange(len(pred)).tolist(), pred-y_test.values.ravel())
    plt.plot(np.arange(len(pred)).tolist(), np.zeros(len(pred)), color="r")

    error_tr = mean_squared_error(pred_tr, y_train)
    error = mean_squared_error(pred, y_test)
    print(error_tr, error)
    
    
def evaluate_(model, x_train, x_test, y_train, y_test):
    # sklearn models
    pred_tr = model.predict(x_train).ravel()
    pred = model.predict(x_test).ravel()
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("train")
    plt.scatter(np.arange(len(pred_tr)).tolist(), pred_tr-y_train.values.ravel())
    plt.plot(np.arange(len(pred_tr)).tolist(), np.zeros(len(pred_tr)), color="r")
    plt.subplot(1, 2, 2)
    plt.title("test")
    plt.scatter(np.arange(len(pred)).tolist(), pred-y_test.values.ravel())
    plt.plot(np.arange(len(pred)).tolist(), np.zeros(len(pred)), color="r")

    error_tr = mean_squared_error(pred_tr, y_train)
    error = mean_squared_error(pred, y_test)
    print(error_tr, error)
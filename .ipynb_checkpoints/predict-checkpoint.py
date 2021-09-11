import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np 
from typing import Dict, List, Any, Tuple 
import os, time, pickle, json 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statistics import mean 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from src.utils.load_model import load_net
from src.utils.download import download_csv

def inference(test_csv):
    with torch.no_grad():
        s = MinMaxScaler(feature_range=(0, 1))
        test = test_csv.values
        test = s.fit_transform(test)
        test = test[-30:, :]
        test_tensor = torch.FloatTensor(test).unsqueeze(0)
        net = load_net()
        y = net(test_tensor)
        y = y[0].cpu().numpy().tolist()
        tomorrow = y[-1]
        print(f"明日の感染者数は{tomorrow:.1f}人と予想されます")

def transform(target_csv, hos_csv, death_csv, severe_csv):
    columns = ['Newly confirmed cases', 'Requiring inpatient care',
       'Discharged from hospital or released from treatment',
       'To be confirmed', 'Deaths(Cumulative)', 'Severe cases']
    target = pd.read_csv(target_csv)
    hos = pd.read_csv(hos_csv)
    death = pd.read_csv(death_csv)
    severe = pd.read_csv(severe_csv)

    target = target.groupby("Date").sum().reset_index()
    hos = hos.groupby("Date").sum().reset_index()
    death = death.groupby("Date").sum().reset_index()
    severe = severe.groupby("Date").sum().reset_index()
    hos_t = pd.merge(target, hos, how="outer", left_on="Date", right_on="Date")
    hos_t_d = pd.merge(hos_t, death, how="outer", left_on="Date", right_on="Date")
    df = pd.merge(hos_t_d, severe, how="outer", left_on="Date", right_on="Date")

    df = df.fillna(0)
    df["Date"] = pd.to_datetime(df.Date)
    df = df.groupby("Date").sum()
    df = df[columns]
    return df 

def main():
    # 厚生労働省のオープンデータセットを指定
    # https://www.mhlw.go.jp/stf/covid-19/open-data.html
    download_csv()
    target_csv = "./data/download/target.csv"
    hos_csv = "./data/download/hos.csv"
    death_csv = "./data/download/death.csv"
    severe_csv = "./data/download/severe.csv"
    df = transform(target_csv, hos_csv, death_csv, severe_csv)
    inference(df)
    
main()
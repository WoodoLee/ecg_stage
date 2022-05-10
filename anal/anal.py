import os
import itertools
import time
import random

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from utils.util_utils import *
from utils.pre_data import *
from models.models import *
import scipy
from scipy.fftpack import fft
import pretty_errors

import warnings
warnings.filterwarnings(action='ignore') 

class Config:
    csv_path = 'data.csv'
    seed = 2022
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    
    attn_state_path = '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/attn.pth'
    lstm_state_path = '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/lstm.pth'
    cnn_state_path =  '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/cnn.pth'
    
    attn_logs = '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/attn.csv'
    lstm_logs = '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/lstm.csv'
    cnn_logs  = '/home/wdlee/data2/healthCare/ecg_stage/data/mitbih_with_synthetic/cnn.csv'
    
    train_csv_path = '/home/wdlee/data2/healthCare/ecg_stage/data/ecg_hb/mitbih_train.csv'
    test_csv_path =  '/home/wdlee/data2/healthCare/ecg_stage/data/ecg_hb/mitbih_test.csv'
    

    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = Config()

# df_mitbih_train = pd.read_csv(config.train_csv_path, header=None)
# df_mitbih_test = pd.read_csv(config.test_csv_path, header=None)
# df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)

# df_mitbih.rename(columns={187: 'class'}, inplace=True)

id_to_label = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}
# df_mitbih['label'] = df_mitbih.iloc[:, -1].map(id_to_label)
# print(df_mitbih.info())

# df_mitbih.to_csv('data.csv', index=False)
# config.csv_path = 'data.csv'

df_mitbih_new = pd.read_csv(config.csv_path)

print(df_mitbih_new)


N = 5
n = 5
samples = [df_mitbih_new.loc[df_mitbih_new['class'] == cls].sample(5) for cls in range(N)]
titles = [id_to_label[cls] for cls in range(5)]

with plt.style.context("seaborn-white"):
    fig, axs = plt.subplots(3, 2, figsize=(20, 7))
    for i in range(5):
        ax = axs.flat[i]
        ax.plot(samples[i].values[:,:-2].transpose())
        ax.set_title(titles[i])
        #plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")

    plt.savefig(f"signals_per_class_5.png", 
                    format="png",bbox_inches='tight', pad_inches=0.2) 



with plt.style.context("seaborn-white"):
    fig, axs = plt.subplots(3, 2, figsize=(20, 7))
    for i in range(5):
        ax = axs.flat[i]
        x_len = samples[i].values[:,:-2].shape[1]
        T = 1.0/125.0 * x_len
        x_f = np.linspace(0.0, 1.0 / T , x_len)
        # print(samples[i].values[:,:-2].transpose())
        fft_sample = scipy.fft.fft(samples[i].values[:,:-2])
        ax.plot(x_f, np.abs(fft_sample.transpose()))
        ax.set_title(titles[i])
        #plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.suptitle("ECG Signals_fft", fontsize=20, y=1.05, weight="bold")
        
    plt.savefig(f"signals_per_class_fft_5.png", 
                    format="png",bbox_inches='tight', pad_inches=0.2) 




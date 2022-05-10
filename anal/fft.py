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
    

config = Config()

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


cls =0 
N=5
sample_n = 5

sample_t = [df_mitbih_new.loc[df_mitbih_new['class'] == cls].sample(sample_n) for cls in range(N)]

dic_fft=dict()
dic_t=dict()
for cls in range(N):
  np_temp = 'np_fft_{}'.format(cls)
  np_temp_t = 'np_t_{}'.format(cls)
  # dic_fft[np_temp] = np.abs(scipy.fft.fft(df_mitbih_new.loc[df_mitbih_new['class'] == cls].sample(sample_n).values[:,:-2]))
  dic_fft[np_temp] = np.abs(scipy.fft.fft(df_mitbih_new.loc[df_mitbih_new['class'] == cls].values[:,:-2])).transpose()
  dic_t[np_temp_t] = df_mitbih_new.loc[df_mitbih_new['class'] == cls].values[:,:-2].transpose()
print(dic_fft.keys())






for cls in list(dic_fft.keys()):
  print(len(dic_fft[cls]))

titles = [id_to_label[cls] for cls in range(5)]
print(titles)



# with plt.style.context("seaborn-white"):
fig, axs = plt.subplots(3, 2, figsize=(20, 7))
for i, key in enumerate(list(dic_fft.keys())):
  ax = axs.flat[i]
  fft_temp = dic_fft[key]
  ax.hist(fft_temp, bins=30, range = [0, 10])
  # for j in range(len(fft_temp)):
    # ax.hist(j, cumulative=True, bins = 50)
  ax.set_title(titles[i])
  #plt.ylabel("Amplitude")
  plt.tight_layout()
  plt.suptitle("ECG Signals_fft", fontsize=20, y=1.05, weight="bold")        
  plt.savefig(f"signals_per_class_fft_hist_total.png", 
                format="png",bbox_inches='tight', pad_inches=0.2) 



# # with plt.style.context("seaborn-white"):
# fig, axs = plt.subplots(3, 2, figsize=(20, 7))
# for i, key in enumerate(list(dic_t.keys())):
#   ax = axs.flat[i]
#   t_temp = dic_t[key]
#   ax.plot(t_temp)
#   # ax.hist(fft_temp, bins=40, range = [0, 10])
#   # for j in range(len(fft_temp)):
#     # ax.hist(j, cumulative=True, bins = 50)
#   ax.set_title(titles[i])
#   #plt.ylabel("Amplitude")
#   plt.tight_layout()
#   plt.suptitle("ECG Signals_t", fontsize=20, y=1.05, weight="bold")        
#   plt.savefig(f"signals_per_class.png", 
#                 format="png",bbox_inches='tight', pad_inches=0.2) 


# with plt.style.context("seaborn-white"):
#     fig, axs = plt.subplots(3, 2, figsize=(20, 7))
#     for i, key in enumerate(list(dic_fft.keys())):
#         fft_temp = dic_fft[key]
#         for j in range(5):
#           fft_temp_1 = fft_temp[j]
#           x_len = len(fft_temp_1)
#           T = 1.0/125.0 * x_len
#           x_f = np.linspace(0.0, 1.0 / T , x_len)
#           ax = axs.flat[i]
#           ax.plot(x_f, fft_temp.transpose())
#           ax.set_title(titles[i])
#         #plt.ylabel("Amplitude")
    
#     plt.tight_layout()
#     plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")
#     plt.savefig(f"signals_per_class_fft_test.svg",
#                     format="svg",bbox_inches='tight', pad_inches=0.2)
        
#     plt.savefig(f"signals_per_class_fft_test.png", 
#                     format="png",bbox_inches='tight', pad_inches=0.2) 


# with plt.style.context("seaborn-white"):
#     fig, axs = plt.subplots(3, 2, figsize=(20, 7))
#     for i in range(5):
#         ax = axs.flat[i]
#         ax.plot(sample_t[i].values[:,:-2].transpose())
#         ax.set_title(titles[i])
#         #plt.ylabel("Amplitude")
    
#     plt.tight_layout()
#     plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")
#     plt.savefig(f"signals_per_class_test.svg",
#                     format="svg",bbox_inches='tight', pad_inches=0.2)
        
#     plt.savefig(f"signals_per_class_test.png", 
#                     format="png",bbox_inches='tight', pad_inches=0.2) 

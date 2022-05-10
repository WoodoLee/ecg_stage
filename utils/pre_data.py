import os
import itertools
import time
import random

import numpy as np
import pandas as pd 

  

def func_pre(config, id_to_label):

  df_mitbih_train = pd.read_csv(config.train_csv_path, header=None)
  df_mitbih_test  = pd.read_csv(config.test_csv_path , header=None)
  
  
  df_mitbih_train.rename(columns={187: 'class'}, inplace=True)
  df_mitbih_test.rename(columns={187: 'class'}, inplace=True)

  df_mitbih_train['label'] = df_mitbih_train.iloc[:, -1].map(id_to_label)
  df_mitbih_test['label'] = df_mitbih_test.iloc[:, -1].map(id_to_label)

  df_mitbih_train.to_csv('./pre_data_train.csv', index=False)
  df_mitbih_test.to_csv('./pre_data_test.csv', index=False)
 
  
  return df_mitbih_train, df_mitbih_test

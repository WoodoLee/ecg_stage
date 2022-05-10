import torch.onnx
from torchviz import make_dot

import os
import itertools
import time
import random

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ExponentialLR

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score


import pretty_errors
from utils.util_utils import *
from utils.pre_data import *
from models.models import *

import torch.onnx as torch_onnx
from torchvision import models
from torchsummary import summary

import hiddenlayer as hl
 
import warnings
warnings.filterwarnings(action='ignore') 

class Config:

    seed = 2022
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    # train_csv_path    = './data/ecg_hb/mitbih_train.csv'
    train_csv_path    = './data/mitbih_with_synthetic/mitbih_with_syntetic_train.csv'
    # test_csv_path     = './data/ecg_hb/mitbih_test.csv'
    test_csv_path     = './data/mitbih_with_synthetic/mitbih_with_syntetic_test.csv'
    results_path      =  './results'

    # pre_csv_train_path = './data/mitbih_with_synthetic/mitbih_with_syntetic_train.csv'
    # pre_csv_test_path =  './data/mitbih_with_synthetic/mitbih_with_syntetic_test.csv'

    attn_logs         = './results/logs/attn.csv'
    lstm_logs         = './results/logs/lstm.csv'
    cnn_logs          = './results/logs/cnn.csv'
    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = Config()

id_to_label = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}

seed_everything(config.seed)


cnn_model = CNN(num_classes=5, hid_size=128).to(config.device)
cnn_model.eval()

lstm_model = RecurrentModel(1, 64, 'lstm', True).to(config.device)

lstm_model.eval()


attn_model = RecurrentAttentionModel(1, 64, 'lstm', False).to(config.device)

attn_model.eval()



models = [cnn_model, lstm_model, attn_model]

print(summary(cnn_model, input_size=(1, 128)))
# params =cnn_model.state_dict()


# dummy_data = torch.empty(1,124).to(config.device)
# torch.onnx.export(cnn_model, dummy_data, "cnn_model.onnx")


 
x = torch.zeros(96, 1, 187).to(config.device) # dummy input
# make_dot(cnn_model(x), params=dict(list(cnn_model.named_parameters()))).render("cnn_torchviz", format="png")
make_dot(cnn_model(x), params=dict(list(cnn_model.named_parameters())), show_attrs=True, show_saved=True).render("./figs/cnn_model", format="png")
make_dot(lstm_model(x), params=dict(list(cnn_model.named_parameters())), show_attrs=True, show_saved=True).render("./figs/lstm_model", format="png")
make_dot(attn_model(x), params=dict(list(attn_model.named_parameters())), show_attrs=True, show_saved=True).render("./figs/attn_model", format="png")


transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
graph = hl.build_graph(cnn_model, x, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('./figs/cnn_hiddenlayer', format='png')
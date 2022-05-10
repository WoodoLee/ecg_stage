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
from utils.util_utils import *
from utils.pre_data import *
from models.models import *
import torch.onnx as torch_onnx
from torchvision import models
from torchsummary import summary
import argparse

import pretty_errors
import warnings
warnings.filterwarnings(action='ignore') 


parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--train_data', type=str, default='../data/mitbih_with_synthetic/mitbih_with_syntetic_train.csv',  help='train csv path')
parser.add_argument('--test_data', type=str, default='../data/mitbih_with_synthetic/mitbih_with_syntetic_test.csv', help='test csv path')
parser.add_argument('--results', type=str, default='./results', help='results path')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu' , help='GPU or CPU') 
parser.add_argument('--model', type=str, default= 'cnn+lstm', help='select the models : cnn, cnn+lstm, cnn+lstm+att ') # 
# parser.add_argument('--model_load', type=str, default= './models/pre_train/cnn.pth', help='train model path')
parser.add_argument('--model_load', type=str, default= './results/pths/best_cnn_model_epoch4.pth', help='train model path')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=1, help='epoch number')
parser.add_argument('--seed', type=int, default=2002, help='seed number')


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = parser.parse_args()

id_to_label = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}

seed_everything(config.seed)

def func_model(config):
    if config.model == 'cnn' : 
        model = CNN(num_classes=5, hid_size=128)
    elif config.model == 'cnn+lstm' : 
        model = RecurrentModel(1, 64, 'lstm', True)
    elif config.model == 'cnn+lstm+attn' : 
        model = RecurrentAttentionModel(1, 64, 'lstm', False)
    return model
    
model = func_model(config)



"""

Train & Validation

"""


if config.phase == 'train':

    trainer = Trainer(config, net=model, lr=1e-3, batch_size = config.batch_size, num_epochs = config.epochs, model_type =  config.model) #100)
    trainer.run()
    train_logs = trainer.train_df_logs
    train_logs.columns = ["train_"+ colname for colname in train_logs.columns]

    val_logs = trainer.val_df_logs
    val_logs.columns = ["val_"+ colname for colname in val_logs.columns]

    logs = pd.concat([train_logs,val_logs], axis=1)
    logs.reset_index(drop=True, inplace=True)
    logs = logs.loc[:, [
        'train_loss', 'val_loss', 
        'train_accuracy', 'val_accuracy', 
        'train_f1', 'val_f1',
        'train_precision', 'val_precision',
        'train_recall', 'val_recall']
                                    ]
    logs.head()
    logs.to_csv(config.results + f"/logs/{config.model}.csv", index=False)


elif config.phase == 'test':

    model.load_state_dict( torch.load(config.model_load, map_location=config.device))
    model.eval()

    test_df = pd.read_csv(config.test_data)
    test_dataset = ECGDataset(test_df)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=96, num_workers=0, shuffle=False)


    y_pred, y_true = func_tester(config, test_dataloader, model)
    y_pred.shape, y_true.shape


    report = pd.DataFrame(
        classification_report(
            y_pred,
            y_true,
            output_dict=True
        )
    ).transpose()

    print(report)

    clf_report = classification_report(y_pred, 
                                    y_true,
                                    labels=[0,1,2,3,4],
                                    target_names=list(id_to_label.values()),#['N', 'S', 'V', 'F', 'Q'],
                                    output_dict=True)


    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=12, rotation=0)


    plt.title("Ensemble Classification Report", fontsize=20)
    plt.savefig(f"./results/figs/{config.model}_result.png", format="png",bbox_inches='tight', pad_inches=0.2)

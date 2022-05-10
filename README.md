# ecg_stage

This repository is for classifying ECG heartbeat categories at Korea University.
There are three models to solve the Task : 5 class classification with ECG signals


1. 1-D CNN 
2. 1-D CNN + LSTM 
3. 1-D CNN + LSTM + Attention 

The details of all models with data flow are shown in ./figs.

Class

0: "Normal"
1: "Artial Premature"
2: "Premature ventricular contraction"
3: "Fusion of ventricular and normal"
4: "Fusion of paced and normal"


## Envs

All models in this repository are tested with

python = 3.9

pyTorch = 1.1 with cuda 11.3

### Conda (env name : ecg)

```
conda env create --file ecg_environment.yml

```


### pip 

```
pip install -r requirement.txt
```

## Structure

```
- data
  - mitbih_with_synthetic
  - ecg_hb
- models
  - pre_train
- results
  - figs
  - logs
  - pths
- anal
- utils

```
## Dataset
| DataSet                                         | Description                                                        | 
| :--------------------------------------------:  | :----------------------------------------------------------------: | 
| mitbih_with_synthetic                           | https://www.kaggle.com/datasets/polomarco/mitbih-with-synthetic    | 
| ecg_hb(ECG Heartbeat Categorization Dataset)    | https://www.kaggle.com/datasets/shayanfazeli/heartbeat             | 

You can download all dataset, which should be unziped in data folder.
All trained models are saved in ./results/pths, and train logs are saved in ./results/logs with csv format. 
Figures in ./results/figs are about the evaluation results - confusion matrix.
Pre-trained models are in ./models/pre_train folder. 
## QuickStart

You can select pheses for models, which are "train" and "test (evaluation)" in the main.py script.

### Arguments

```bash
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--train_data', type=str, default='../data/mitbih_with_synthetic/mitbih_with_syntetic_train.csv',  help='train csv path')
parser.add_argument('--test_data', type=str, default='../data/mitbih_with_synthetic/mitbih_with_syntetic_test.csv', help='test csv path')
parser.add_argument('--results', type=str, default='./results', help='results path')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu' , help='GPU or CPU') 
parser.add_argument('--model', type=str, default= 'cnn+lstm', help='select the models : cnn, cnn+lstm, cnn+lstm+attn ')
parser.add_argument('--model_load', type=str, default= './results/pths/best_cnn_model_epoch4.pth', help='train model path')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=1, help='epoch number')
parser.add_argument('--seed', type=int, default=2002, help='seed number')

```

### Train 
```
python main.py --phase train

```
### Evaluation (Test)
```
python main.py --phase test

```
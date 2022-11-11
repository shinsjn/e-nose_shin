# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 07:05:48 2021

@author: Jb
"""
from glob import glob
import pandas as pd
import numpy as np
import random
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

Acetaldehyde_L1 = glob('Acetaldehyde_500/L1/*')
Acetone_L1 = glob('Acetone_2500/L1/*')
Ammonia_L1 = glob('Ammonia_10000/L1/*')
Benzene_L1 = glob('Benzene_200/L1/*')
Butanol_L1 = glob('Butanol_100/L1/*')
CO1000_L1 = glob('CO_1000/L1/*')
CO4000_L1 = glob('CO_4000/L1/*')
Ethylene_L1 = glob('Ethylene_500/L1/*')
Methane_L1 = glob('Methane_1000/L1/*')
Methanol_L1 = glob('Methanol_200/L1/*')
Toluene_L1 = glob('Toluene_200/L1/*')


paths = Acetaldehyde_L1 + Acetone_L1 + Ammonia_L1 + Benzene_L1 + Butanol_L1 + CO1000_L1 + CO4000_L1 + Ethylene_L1 + Methane_L1 + Methanol_L1 + Toluene_L1

# selectN = np.Inf
# for i, path in enumerate(paths):
#     df = pd.read_csv(path, sep='\t', header=None)
#     frequency = df.values[:,0]
#     st = len(frequency[frequency < 20000])
#     frequency = frequency[st:]
#     # resistance = resistance[st:]
#     if selectN > frequency.shape[0]:
#         selectN = frequency.shape[0]
selectN = 9248

AcetaldehydeR = np.zeros((len(Acetaldehyde_L1), selectN))
AcetaldehydeF = np.zeros((len(Acetaldehyde_L1), selectN))
for i, path in enumerate(Acetaldehyde_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    AcetaldehydeR[i] = np.take(resistance, randIndex)
    AcetaldehydeF[i] = np.take(frequency, randIndex)
    
AcetoneR = np.zeros((len(Acetone_L1), selectN))
AcetoneF = np.zeros((len(Acetone_L1), selectN))
for i, path in enumerate(Acetone_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    AcetoneR[i] = np.take(resistance, randIndex)
    AcetoneF[i] = np.take(frequency, randIndex)
    
AmmoniaR = np.zeros((len(Ammonia_L1), selectN))
AmmoniaF = np.zeros((len(Ammonia_L1), selectN))
for i, path in enumerate(Ammonia_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    AmmoniaR[i] = np.take(resistance, randIndex)
    AmmoniaF[i] = np.take(frequency, randIndex)
        
BenzeneR = np.zeros((len(Benzene_L1), selectN))
BenzeneF = np.zeros((len(Benzene_L1), selectN))
for i, path in enumerate(Benzene_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    BenzeneR[i] = np.take(resistance, randIndex)
    BenzeneF[i] = np.take(frequency, randIndex)
    
ButanolR = np.zeros((len(Butanol_L1), selectN))
ButanolF = np.zeros((len(Butanol_L1), selectN))
for i, path in enumerate(Butanol_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    ButanolR[i] = np.take(resistance, randIndex)
    ButanolF[i] = np.take(frequency, randIndex)

CO1000R = np.zeros((len(CO1000_L1), selectN))
CO1000F = np.zeros((len(CO1000_L1), selectN))
for i, path in enumerate(CO1000_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    CO1000R[i] = np.take(resistance, randIndex)
    CO1000F[i] = np.take(frequency, randIndex)
    
CO4000R = np.zeros((len(CO4000_L1), selectN))
CO4000F = np.zeros((len(CO4000_L1), selectN))
for i, path in enumerate(CO4000_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    CO4000R[i] = np.take(resistance, randIndex)
    CO4000F[i] = np.take(frequency, randIndex)
    

EthyleneR = np.zeros((len(Ethylene_L1), selectN))
EthyleneF = np.zeros((len(Ethylene_L1), selectN))
for i, path in enumerate(Ethylene_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    EthyleneR[i] = np.take(resistance, randIndex)
    EthyleneF[i] = np.take(frequency, randIndex)
    
MethaneR = np.zeros((len(Methane_L1), selectN))
MethaneF = np.zeros((len(Methane_L1), selectN))
for i, path in enumerate(Methane_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    MethaneR[i] = np.take(resistance, randIndex)
    MethaneF[i] = np.take(frequency, randIndex)
    
MethanolR = np.zeros((len(Methanol_L1), selectN))
MethanolF = np.zeros((len(Methanol_L1), selectN))
for i, path in enumerate(Methanol_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    MethanolR[i] = np.take(resistance, randIndex)
    MethanolF[i] = np.take(frequency, randIndex)

TolueneR = np.zeros((len(Toluene_L1), selectN))
TolueneF = np.zeros((len(Toluene_L1), selectN))
for i, path in enumerate(Toluene_L1):
    df = pd.read_csv(path, sep='\t', header=None)
    frequency = df.values[:,0]
    array5 = df.values[:, 48:56]
    resistance = array5[:, 3]
    st = len(frequency[frequency < 20000])
    frequency = frequency[st:]
    resistance = resistance[st:]
    randIndex = random.sample(range(len(frequency)), selectN)
    randIndex.sort()
    TolueneR[i] = np.take(resistance, randIndex)
    TolueneF[i] = np.take(frequency, randIndex)

x = np.vstack((AcetaldehydeR, AcetoneR, AmmoniaR, BenzeneR, ButanolR, 
               CO4000R, EthyleneR, MethaneR, MethanolR, TolueneR))
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
x = x.reshape(x.shape[0], 1, x.shape[1])
y = np.array([0]*len(Acetaldehyde_L1)+[1]*len(Acetone_L1)+[2]*len(Ammonia_L1)+ 
             [3]*len(Benzene_L1)+[4]*len(Butanol_L1)+[5]*len(CO4000_L1)+
             [6]*len(Ethylene_L1)+[7]*len(Methane_L1)+[8]*len(Methanol_L1)+[9]*len(Toluene_L1))
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y).type(torch.LongTensor)
dataset = torch.utils.data.TensorDataset(x, y)
torch.save(dataset, 'uci_tensor.pt')
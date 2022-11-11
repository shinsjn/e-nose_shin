# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:07:35 2021

@author: Jb
"""

import pandas as pd
import numpy as np
import torch 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from resnet1d import ResNet1D
from tqdm import trange
from time import sleep
from sklearn import preprocessing
import seaborn as sn

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
     
x = np.load('modified_x.npy')
y = np.load('modified_y.npy')

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
x = x.reshape(x.shape[0], 1, x.shape[1])
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y).type(torch.LongTensor)
dataset = torch.utils.data.TensorDataset(x, y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 4
k_folds = 4
batch_size = 4
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
learning_rate = 0.0001
epochs = 600
criterion = torch.nn.CrossEntropyLoss()
results = {}
trues = []
preds = []
for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)): 
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    sleep(0.1)
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=dataset.__len__(), sampler=test_subsampler)
    # model = torch.load('savedmodel')
    
    model = torch.load('resnet1d.pt') # import pretrained model
    # freeze all layers except the last layer
    # this is for fine tuning the classifier layer
    for param in model.parameters():
        param.requires_grad = False
    infeatures = model.dense.in_features
    model.dense = torch.nn.Linear(infeatures, num_classes)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    tepoch = trange(epochs, desc="Epochs")
    for epoch in tepoch:
        tmploss_list = []
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)   
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item()) 
        if epoch % 5 == 0:
            # print("Loss:", loss.item())
            tepoch.set_postfix(loss=loss.item())
    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():
      # Iterate over the test data and generate predictions
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)  

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            trues += list(targets.to('cpu').numpy())
            preds += list(predicted.to('cpu').numpy())
            # print('\nActual   :', targets)
            # print('Predicted:', predicted)
            print('total:', total, ' Correct:', correct)
        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
    # step = np.linspace(0, epochs, epochs)
    # plt.plot(step, np.array(loss_list))
    # plt.xlabel('epoch')
    # plt.ylabel('loss')

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
  print(f'Fold {key}: {value} %')
  sum += value
print(f'Average: {sum/len(results.items())} %')  
# step = np.linspace(0, epochs, epochs)
# plt.savefig('loss_graph_resnet50-1.png')
# plt.show
# print('Time:', end - start)
# plt.plot(step, np.array(loss_list))
# plt.xlabel('epoch')
# plt.ylabel('loss')

# print confusion matrix
conf_matrix = confusion_matrix(trues, preds)
class_report = classification_report(trues, preds, target_names=['nh3', 'no2', 'mixture1', 'mixture2'])
# class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(1)


ax = plt.subplot()
labels = ["NH3", "NO2", "Mixture1", "Mixture2"]
df_cm = pd.DataFrame(conf_matrix, index=labels,columns=labels)
# sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show
ax.figure.savefig('conf_matrix.png')
print()
print("Class Report:".center(50))
print(class_report)

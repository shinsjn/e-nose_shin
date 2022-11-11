# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 00:04:05 2021

@author: Jb
"""

import numpy as np
import torch 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from resnet1d import ResNet1D 
from time import sleep
from tqdm import trange

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

temp = np.loadtxt('./concatdata.dat')

x = temp[:,:-1]
y = temp[:,-1]
print(x.shape)
print(y.shape)

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
x = x.reshape(x.shape[0], 1, x.shape[1])
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y).type(torch.LongTensor)
dataset = torch.utils.data.TensorDataset(x, y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
k_folds = 4
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
batch_size = 16
epochs = 500
criterion = torch.nn.CrossEntropyLoss()
results = {}

best_lr = -1
best_acc = -1
for lr in [0.001,0.0001,0.00001,0.000001]:
    learning_rate = lr

    for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)):
        # Print%
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=test_subsampler)
        kernel_size = 16
        stride = 2
        n_block = 48
        downsample_gap = 6
        increasefilter_gap = 12
        model = ResNet1D(
            in_channels=1,
            base_filters=64, # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=kernel_size,
            stride=stride,
            groups=32,
            n_block=n_block,
            n_classes=7,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_do=True)
        model.to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_list = []
        for epoch in trange(epochs):
            tmploss_list = []
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_list.append(loss.item())
            print(loss.item())
            if epoch % 100 == 0:
                print("Epoch:", epoch, " Loss:", loss.item())
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
                # print('Actual   :', targets)
                # print('Predicted:', predicted)
            print('total:', total, ' Correct:', correct)
            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
        step = np.linspace(0, epochs, epochs)
        plt.plot(step, np.array(loss_list))
        plt.xlabel('epoch')
        plt.ylabel('loss')
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
      print(f'Fold {key}: {value} %')
      sum += value
    print(f'Average: {sum/len(results.items())} %')
    step = np.linspace(0, epochs, epochs)

    #torch.save(model.state_dict(), 'resnet1d_starte_dict.pt')
    #torch.save(model, 'resnet1d.pt')

    if best_acc < sum/len(results.items()):
        best_acc=sum/len(results.items())
        best_lr=lr

print('best_acc: ',best_acc)
print('best_lr: ', best_lr)
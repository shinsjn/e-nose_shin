# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:29:24 2021

@author: Jb
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from cnn1d import CNN_1d
from tqdm import trange

best_lr = -1
best_acc = -1
for lr in [0.001,0.0001,0.00001,0.000001]:
    #temp = np.loadtxt('./data/NEW_e_nose_data_2022.dat')
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
    # torch.save(dataset, 'tensor.pt')

    # dataset = torch.load('tensor.pt', map_location=lambda storage, loc: storage)
    def reset_weights(m):
      for layer in m.children():
       if hasattr(layer, 'reset_parameters'):
        # print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_folds = 4
    batch_size = 32
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    learning_rate = lr
    epochs = 300
    criterion = torch.nn.CrossEntropyLoss()
    results = {}
    import time
    start = time.time()
    for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)):
        # Print
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
                          batch_size=dataset.__len__(), sampler=test_subsampler)
        model = CNN_1d(channel1=50, channel2=100, lin1=250, out_size=7, ker_size=9, pool_size=2)
        model.to(device)
        model.apply(reset_weights)
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
            if epoch % 20 == 0:
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
                print('\nActual   :', targets)
                print('Predicted:', predicted)
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
    if best_acc < sum/len(results.items()):
        best_acc=sum/len(results.items())
        best_lr=lr
    step = np.linspace(0, epochs, epochs)
    end = time.time()
    elapsed = round(end - start)
    # print('Time:', end - start)
    # plt.plot(step, np.array(loss_list))
    # plt.xlabel('epoch')
    # plt.ylabel('loss')

print('best_acc: ',best_acc)
print('best_lr: ', best_lr)
import torch
import torch.nn as nn
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

def print_report(labels, true, predicted):
    class_report = classification_report(true, predicted, target_names=labels)
    conf_matrix = confusion_matrix(true, predicted)
    print('\nClass Report: \n', class_report)
    plt.figure(2)
    ax = plt.subplot()
    df_cm = pd.DataFrame(conf_matrix, index=labels,columns=labels)
    # sn.set(font_scale=1.0)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()
    
class AE(nn.Module):
    def __init__(self, input_dim, hid1, hid2, hid3, output_dim):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hid1),
            nn.ReLU(),
            nn.Linear(hid1, hid2),
            nn.ReLU(),
            nn.Linear(hid2,hid3),
            nn.ReLU(),
            nn.Linear(hid3,output_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(output_dim, hid3),
            nn.ReLU(),
            nn.Linear(hid3, hid2),
            nn.ReLU(),
            nn.Linear(hid2, hid1),
            nn.ReLU(),
            nn.Linear(hid1,input_dim)
        )
        self.loss = nn.MSELoss()

    def forward(self,x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return encoded, decoded
    def calculate_loss(self,x,decoded):
        return self.loss(x,decoded)

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
    
if __name__ == '__main__':
    x = np.load('modified_x.npy')
    y = np.load('modified_y.npy')
    x = StandardScaler().fit_transform(x)
    # clf = RandomForestClassifier()
    clf = DecisionTreeClassifier()
    
    printicipal_components = 2
    verbose = True
    avg_list = []
    labels = ['NH3', 'NO2', 'Mixture1', 'Mixture2']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(10):
        wrongs = []
        true2 = []
        pred2 = []
        x2 = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            k_folds = 4
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
            true = []
            predicted = []
            scores = []
            for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)): 
                # tmp_ids = train_ids
                # train_ids = test_ids
                # test_ids = tmp_ids
                
                x_train = x[train_ids]
                x_test = x[test_ids]
                x2 = x[test_ids]
                y_train = y[train_ids]
                y_test = y[test_ids]
                # Autoencoder
                model = AE(input_dim=x.shape[1], hid1=1024, hid2=512, hid3=512, output_dim=2)
                model.to(device)
                model.apply(reset_weights)
                optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
                data_x = torch.FloatTensor(x_train)
                data_x_test = torch.FloatTensor(x_test)
                # train
                epoch = 50
                pbar = tqdm(desc = 'training...',total=epoch,position=0)    
                for EPOCH in range(1,epoch+1):
                    ds = torch.utils.data.DataLoader(data_x,batch_size=8,shuffle=True)
                    for tr in ds:
                        tr = tr.to(device)
                        optimizer.zero_grad()
                        en, de = model(tr)
                        loss = model.calculate_loss(tr,de)
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix({'loss':loss.item()})
                    pbar.update(1)
                pbar.close()
                x_, _ = model(data_x.to(device))
                x_ = x_.cpu().detach().numpy()
    
                x_test ,_ = model(data_x_test.to(device))
                x_test = x_test.cpu().detach().numpy()
                # clf.fit(x_train, y_train)
                clf.fit(x_, y_train)
                # print('Fold ', fold, ': ', clf.score(x_test, y_test)*100, '%\n', end='')
                if (verbose):
                    print('Fold ', fold, ': ', clf.score(x_test, y_test)*100, '%\n', end='')
    
                scores.append(clf.score(x_test, y_test))
                true += list(y_test)
                true2 = list(y_test)
                predicted += list(clf.predict(x_test))
                pred2 = list(clf.predict(x_test))
                for i in range(len(true2)):
                    if (true2[i] != pred2[i]):
                        wrongs.append(x2[i])
                # print('pred label:', clf.predict(x_test))
    
            fold_avg = np.average(scores)*100
            avg_list.append(fold_avg)
            wrongs = np.array(wrongs)
            if (verbose):
                print("Average of the folds:", fold_avg)
                print_report(labels, true, predicted)
    
            # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
            # plt.xlabel("Principal Component1")
            # plt.ylabel("Principal Component2")
            # plt.title('PCA with %d principal components' % (printicipal_components))
    print("Avg: %.2f +- %.2f" % (np.average(avg_list), np.std(avg_list)))
    
    tensor_x = torch.from_numpy(x).float()
    reduced_x, _ = model(tensor_x.to(device))
    reduced_x = reduced_x.detach().cpu().numpy()
    yname = []
    for i in y:
        if i == 0:
            yname.append('NH3')
        elif i == 1:
            yname.append('NO2')
        elif i == 2:
            yname.append('Mixture1')
        elif i == 3:
            yname.append('Mixture2')

    correctornot = []
    for i in x:
        if i in wrongs:
            correctornot.append('Incorrect')
        else:
            correctornot.append('correct')
    
    df = pd.DataFrame()
    df["y"] = yname
    df['encoded-dim1'] = reduced_x[:, 0]
    df['encoded-dim2'] = reduced_x[:, 1]
    df['correct'] = correctornot
    lbs = ['NH3', 'NO2', 'Mixture1', 'Mixture2', 'Incorrect']
    sn.scatterplot(x="encoded-dim1", y="encoded-dim2", hue=df.y.tolist(), 
                   style='correct', palette=sn.color_palette("hls", 4),
                    data=df).set(title="T-SNE projection")
    
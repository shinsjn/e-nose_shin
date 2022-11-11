#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:22:30 2021

@author: jb
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

x = np.load('modified_x.npy')
y = np.load('modified_y.npy')
x = StandardScaler().fit_transform(x)
labels = ['NH3', 'NO2', 'Mixture1', 'Mixture2']

principal_components = 50
learning_rates=[30,50,100,150,200,250,300,350]
perplexities=[5,10,20,30,40,50]
for learning_rate in learning_rates:
    for perplexity in perplexities:
        pca = PCA(n_components=0.95)
        x = pca.fit_transform(x)

        tsne = TSNE(n_components=2,learning_rate=learning_rate, perplexity=perplexity)
        x_tsne = tsne.fit_transform(x)

        df = pd.DataFrame()
        df["y"] = y
        df['pca-one'] = x_tsne[:, 0]
        df['pca-two'] = x_tsne[:, 1]
        title = "T_SNE projection" + "["+str(learning_rate) +"|"+ str(perplexity) + "]"
        print(title)
        sns.scatterplot(x="pca-one", y="pca-two", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 4),
                        data=df).set(title=title)
        plt.show()

# df2 = pd.DataFrame()
# df2['wrong-x'] = wrongs[:, 0]
# df2['wrong-y'] = wrongs[:, 1]
# sns.scatterplot(x="wrong-x", y="wrong-y", data=df2)

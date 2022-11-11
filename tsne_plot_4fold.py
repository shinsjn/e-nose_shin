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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
temp = np.loadtxt('./concatdata.dat')
#temp = np.loadtxt('./data/2022_mixture_data.dat')
x = temp[:,:-1]
y = temp[:,-1]

x = StandardScaler().fit_transform(x)
labels = ['NO2','NH3','1:1','1:2','1:3','1:4','1:5']

principal_components = 50
learning_rates=[10,20,30,50,100,150]
perplexities=[1,2,3,5,10,]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    skf = StratifiedKFold(n_splits=4, shuffle=True)

    k_folds = 4
    for learning_rate in learning_rates:
        for perplexity in perplexities:
            for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)):

                x_train = x[train_ids]
                x_test = x[test_ids]
                y_train = y[train_ids]
                y_test = y[test_ids]

                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
                lda = LDA(n_components=5,tol=1e-8)
                pca = PCA(n_components=5)
                print('PCA component: ',pca.n_components)
                lda.fit(x_train,y_train)
                pca.fit(x_train)
                #_train = lda.transform(x_train)
                # x_test = lda.transform(x_test)

                print("PCA:", pca.explained_variance_ratio_)
                print("LDA:", lda.explained_variance_ratio_)

                tsne = TSNE(n_components=1, learning_rate=learning_rate, perplexity=perplexity)
                x_tsne = tsne.fit_transform(x_test)

                df = pd.DataFrame()
                df["y"] = y_test
                df['pca-one'] = x_tsne[:, 0]
                #df['pca-two'] = x_tsne[:, 1]
                title = "T_SNE projection" + "[" + str(fold) + "&" + str(learning_rate) + "&" + str(perplexity) + "]"
                y_list = df.y.tolist()
                y_list = set(y_list)
                y_list = list(y_list)
                #sns.scatterplot(x="pca-one", y="pca-two", hue=df.y.tolist(),palette=sns.color_palette("hls", 7),data=df).set(title=title)

                plt.scatter(x = "y", y = "pca-one",c='royalblue', data=df)
                plt.legend()
                plt.xlabel('Label: NO2,NH3,1:1 ~ 1:5 Mixture')
                plt.ylabel('pca_value')
                plt.savefig('C:/Users/shins/Desktop/MLPA/E-nose/회의록/result/1108_result/tsne_1D/'+title+".PNG")
                plt.show()

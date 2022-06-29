import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import time
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import interp
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

import pickle

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)


def fingerprint_distance(f1,f2):
	f_shape=[30,20]
	f1_cut=f1.reshape(f_shape)[~np.all(f1.reshape(f_shape) == 0, axis=1)]
	f2_cut=f2.reshape(f_shape)[~np.all(f2.reshape(f_shape) == 0, axis=1)]
	if f1_cut.shape[0]!=f2_cut.shape[0]:
		region=np.min([f1_cut.shape[0],f2_cut.shape[0]])
		f1_cut=f1_cut[(f1_cut.shape[0]-region if f1_cut.shape[0]>region else 0):]
		f2_cut=f2_cut[(f2_cut.shape[0]-region if f2_cut.shape[0]>region else 0):]
	return 1-np.mean(np.sum(f1_cut==f2_cut,axis=1)/f_shape[1])

def fingerprint_similarity(f1,f2):
	f_shape=[30,20]
	f1_cut=f1.reshape(f_shape)[~np.all(f1.reshape(f_shape) == 0, axis=1)]
	f2_cut=f2.reshape(f_shape)[~np.all(f2.reshape(f_shape) == 0, axis=1)]
	if f1_cut.shape[0]!=f2_cut.shape[0]:
		region=np.min([f1_cut.shape[0],f2_cut.shape[0]])
		f1_cut=f1_cut[(f1_cut.shape[0]-region if f1_cut.shape[0]>region else 0):]
		f2_cut=f2_cut[(f2_cut.shape[0]-region if f2_cut.shape[0]>region else 0):]
	return np.mean(np.sum(f1_cut==f2_cut,axis=1)/f_shape[1])


def fingerprint_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = fingerprint_similarity(x, y)
    return gram_matrix
"""

def fingerprint_kernel(X,Y):
	return pd.concat([pd.DataFrame({i:fing for i,fing in enumerate(X)}),\
					  pd.DataFrame({i:fing for i,fing in enumerate(Y)})],axis=1)\
					.corr(method=fingerprint_similarity).to_numpy()[X.shape[0]:,:-Y.shape[0]].T
"""

#df=pd.read_parquet("./data/other/all_fingerprints_updated.parquet")
#df=pd.read_parquet("./data/other/fingerprints_8_fams.parquet")
df=pd.read_parquet("/storage/out/meloni_samples_5/fingerprints_10_fams.parquet")
df["all_zeros"]=df.fingerprint.apply(lambda x:np.all(x==0))
df=df.loc[df.all_zeros==False]
#df_capped=df.groupby('category').head(530).reset_index(drop=True)
#df_capped=df.groupby('category').head(330).reset_index(drop=True)
df_capped=df.groupby('category').head(190).reset_index(drop=True)
df_capped = df_capped[df_capped.category != "zusy"]
n_neighbors = 10
df = df_capped; n_neighbors = n_neighbors - 1

#x=np.array(df_capped.fingerprint.tolist())
x=np.array(df.fingerprint.tolist())
#y=LabelEncoder().fit_transform(df_capped.category.values)
y=LabelEncoder().fit_transform(df.category.values)
labels=np.unique(y)
target_names=[df.category.values[y==i][0] for i in labels]
f_shape=[30,20]

random_state = np.random.RandomState(0)

# Binarize the output
y = label_binarize(y, classes=labels)
n_classes = y.shape[1]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                    random_state=0)


# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel=fingerprint_kernel, probability=True, random_state=random_state))

# classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors,metric=fingerprint_distance))#,metric_params={"f_shape":f_shape}))

y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


with open(f"/storage/datadump{time.time_ns()}.pkl", "wb") as fb:
	pickle.dump({"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc, "target_names": target_names}, fb)


plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'{target_names[i]} (area = {np.around(roc_auc[i])})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# plt.savefig("/storage/logs/roc_curves_{}.png".format(sys.argv[1]))


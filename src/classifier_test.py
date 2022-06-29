from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")

def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

"""
def fingerprint_distance(f1,f2,f_shape):
	f1_cut=f1[f1!=0].reshape((-1,f_shape[1]))
	f2_cut=f2.reshape((-1,f_shape[1]))[:f1_cut.shape[0]]
	return 1-np.mean(np.sum(f1_cut==f2_cut,axis=1)/f_shape[1])     
"""

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
df=pd.read_parquet("/storage/out/meloni_samples_5/fingerprints_10_fams.parquet")
df["all_zeros"]=df.fingerprint.apply(lambda x:np.all(x==0))
df=df.loc[df.all_zeros==False]
#df_capped=df.groupby('category').head(530).reset_index(drop=True)
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

## CLASSIFICATION
ndict = dict()
for n_neighbors in range(5, 21, 5):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,metric=fingerprint_distance)#,metric_params={"f_shape":f_shape})
        # clf = svm.SVC(kernel=fingerprint_kernel)

        splits=5
        random_state = np.random.RandomState(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        cv = StratifiedKFold(n_splits=splits)

        #accs=[]
        metrics_data=[]

        for i, (train, val) in enumerate(cv.split(x, y)):
                print(f"fold {i+1}:")
                start=time.time()
                clf.fit(x[train],y[train])
                end=time.time()
                print(f"\ttraining:")
                print(f"\t\tsize: {x[train].shape[0]}")
                print(f"\t\telapsed time: {timer(start,end)}")
                start=time.time()
                #acc=clf.score(x[val],y[val])*100
                y_pred=clf.predict(x[val])
                end=time.time()
                print(f"\ttest:")
                print(f"\t\tsize: {x[val].shape[0]}")
                print(f"\t\telapsed time: {timer(start,end)}")
                #print(f"\t\taccuracy: {acc}")
                print("\n")
                #accs.append(acc)
                clfrep = classification_report(y[val],y_pred,labels=labels,target_names=target_names, output_dict=True)
                if n_neighbors not in clfrep.keys():
                        ndict.update({n_neighbors: []})
                ndict[n_neighbors] += [np.array([clfrep["macro avg"]["precision"], clfrep["macro avg"]["recall"], clfrep["macro avg"]["f1-score"]])]
        ndict[n_neighbors] = np.mean(ndict[n_neighbors], axis=0)
        print(ndict[n_neighbors])
print(ndict)



"""
LAST CHECK 98.16037735849056
upatre    530
shipup    530
autoit    530
sivis     530

LAST CHECK 82.79761904761905
upatre         420
shipup         420
autoit         420
sivis          420
gepys          420
ausiv          420
installcore    420
virlock        420

LAST CHECK with cleared data (empty fingerprints) 86.24999999999999
upatre         330
shipup         330
autoit         330
sivis          330
gepys          330
ausiv          330
installcore    330
virlock        330

KNN 9 neighbors
fold 1:
        training:
                size: 2112
                elapsed time: 00:00:01.27
        test:
                size: 528
                elapsed time: 00:01:06.76
                accuracy: 86.74242424242425


fold 2:
        training:
                size: 2112
                elapsed time: 00:00:01.27
        test:
                size: 528
                elapsed time: 00:01:08.39
                accuracy: 82.57575757575758


fold 3:
        training:
                size: 2112
                elapsed time: 00:00:01.26
        test:
                size: 528
                elapsed time: 00:01:05.14
                accuracy: 86.5530303030303


fold 4:
        training:
                size: 2112
                elapsed time: 00:00:01.27
        test:
                size: 528
                elapsed time: 00:01:09.72
                accuracy: 88.63636363636364


fold 5:
        training:
                size: 2112
                elapsed time: 00:00:01.26
        test:
                size: 528
                elapsed time: 00:01:06.13
                accuracy: 87.5


86.40151515151516


WITH SVM
fold 1:
        training:
                size: 2112
                elapsed time: 00:05:39.53
        test:
                size: 528
                elapsed time: 00:01:24.71
                accuracy: 87.31060606060606


fold 2:
        training:
                size: 2112
                elapsed time: 00:05:41.34
        test:
                size: 528
                elapsed time: 00:01:25.56
                accuracy: 87.12121212121212


fold 3:
        training:
                size: 2112
                elapsed time: 00:05:42.02
        test:
                size: 528
                elapsed time: 00:01:25.84
                accuracy: 86.74242424242425


fold 4:
        training:
                size: 2112
                elapsed time: 00:05:45.48
        test:
                size: 528
                elapsed time: 00:01:25.45
                accuracy: 88.63636363636364


fold 5:
        training:
                size: 2112
                elapsed time: 00:05:41.38
        test:
                size: 528
                elapsed time: 00:01:25.70
                accuracy: 87.68939393939394


87.5
"""

"""
## OUTLIERS DETECTION
lof = LocalOutlierFactor(n_neighbors = 5, metric=fingerprint_distance, metric_params={"f_shape":f_shape})
inliers_outliers=lof.fit_predict(x)
print(np.sum(inliers_outliers==1)*100/inliers_outliers.size)
print(np.sum(inliers_outliers==-1)*100/inliers_outliers.size) 
"""

"""
## NOVELTY DETECTION
accs=[]
for label in ["upatre","shipup","autoit","sivis","gepys","ausiv","installcore","virlock"]:
        print(f"family: {label}")
        x_novelty_train=np.array(df.loc[df.category!=label].fingerprint.tolist())
        x_novelty_test=np.array(df.loc[df.category==label].fingerprint.tolist())
        lof = LocalOutlierFactor(n_neighbors = 5, novelty = True, metric=fingerprint_distance)#, metric_params={"f_shape":f_shape})
        start=time.time()
        lof.fit(x_novelty_train)
        end=time.time()
        print(f"\ttraining:")
        print(f"\t\tsize: {x_novelty_train.shape[0]}")
        print(f"\t\telapsed time: {timer(start,end)}")
        start=time.time()
        x_pred=lof.predict(x_novelty_test)
        end=time.time()
        print(f"\ttest:")
        print(f"\t\tsize: {x_novelty_test.shape[0]}")
        print(f"\t\telapsed time: {timer(start,end)}")
        print(f"\t {label} samples detected as novelty: {np.sum(x_pred==-1)*100/x_pred.size}")
        accs.append(np.sum(x_pred==-1)*100/x_pred.size)
print(np.array(accs).mean())
"""
#67.196
"""
#print(f"family: {label}")
x_novelty_train=np.array(df.loc[df.category.isin(["upatre","shipup","autoit","gepys","installcore","virlock"])].fingerprint.tolist())
x_novelty_test=np.array(df.loc[df.category=="ausiv"].fingerprint.tolist())
lof = LocalOutlierFactor(n_neighbors = 5, novelty = True, metric=fingerprint_distance)#, metric_params={"f_shape":f_shape})
start=time.time()
lof.fit(x_novelty_train)
end=time.time()
print(f"\ttraining:")
print(f"\t\tsize: {x_novelty_train.shape[0]}")
print(f"\t\telapsed time: {timer(start,end)}")
start=time.time()
x_pred=lof.predict(x_novelty_test)
end=time.time()
print(f"\ttest:")
print(f"\t\tsize: {x_novelty_test.shape[0]}")
print(f"\t\telapsed time: {timer(start,end)}")
print(f"\tausiv samples detected as novelty: {np.sum(x_pred==-1)*100/x_pred.size}")
accs.append(np.sum(x_pred==-1)*100/x_pred.size)
"""
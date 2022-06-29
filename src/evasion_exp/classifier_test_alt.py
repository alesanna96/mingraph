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
import os
import sys

warnings.filterwarnings("ignore")

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

def main_classification(parquet_path, output_folder, n_neighbors=9):
        parquet_name = os.path.basename(parquet_path)
        sys.stdout = open(os.path.join(output_folder, f"{parquet_name}.txt"), "w")

        print(parquet_name)
        df=pd.read_parquet(parquet_path)
        df["all_zeros"]=df.fingerprint.apply(lambda x:np.all(x==0))
        df=df.loc[df.all_zeros==False]

        #x=np.array(df_capped.fingerprint.tolist())                                                                                                                            
        x=np.array(df.fingerprint.tolist())
        #y=LabelEncoder().fit_transform(df_capped.category.values)
        y=LabelEncoder().fit_transform(df.category.values)
        labels=np.unique(y)                                                                                                     
        target_names=[df.category.values[y==i][0] for i in labels]
        f_shape=[30,20]

        ## CLASSIFICATION
        #clf = KNeighborsClassifier(n_neighbors=n_neighbors,metric=fingerprint_distance)#,metric_params={"f_shape":f_shape})
        clf = svm.SVC(kernel=fingerprint_kernel)

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
                print(classification_report(y[val],y_pred,labels=labels,target_names=target_names))

        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
        parquet_folder = r"/home/fra/Desktop/storage/out/meloni_samples_5/evasion_exps_new_bal/dfout/"
        parquet_files = [os.path.join(parquet_folder, x) for x in os.listdir(parquet_folder)]
        output_path = r"/home/fra/Desktop/storage/out/meloni_samples_5/evasion_exps_new_bal/expout/"
        for pf in parquet_files:
                main_classification(pf, output_path)
                print("done with {}".format(pf))

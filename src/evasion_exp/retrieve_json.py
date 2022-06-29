import pandas as pd
import numpy as np
import os
import shutil as su

if __name__ == "__main__":
    df=pd.read_parquet("/storage/out/meloni_samples_5/fingerprints_10_fams.parquet")
    df["all_zeros"]=df.fingerprint.apply(lambda x:np.all(x==0))
    df=df.loc[df.all_zeros==False]
    df_capped=df.groupby('category').head(190).reset_index(drop=True)
    df_capped = df_capped[df_capped.category != "zusy"]
    n_neighbors = 10
    df = df_capped; n_neighbors = n_neighbors - 1
    p1 = "/storage/out/meloni_samples_5/graphs/"
    p2 = "/home/fra/Documents/thesis/data/graphs/"
    p3 = "/home/fra/Documents/thesis/data/graphs/pack4/"
    graph_list1 = os.listdir(p1)
    graph_list2 = os.listdir(p2)
    graph_list3 = os.listdir(p3)

    ps = [p1, p2, p3]
    graphs = [graph_list1, graph_list2, graph_list3]
    of = r"/storage/out/meloni_samples_5/graphs_balanced/"
    os.mkdir(of)
    to_cp = []

    for x in df.name:
        for i in range(len(ps)):
            p = ps[i]
            graph_listX = graphs[i]
            lx = [os.path.join(p, y) for y in graph_listX if x in y]
            if lx:
                to_cp += lx
                break

    for tc in to_cp:
        su.copy2(tc, of)

    print(1)
import numpy as np
from datasketch import MinHash
import networkx as nx
import pandas as pd
import json
import os
import time

def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def compute_adj_mat(exe_df):
    g=nx.Graph()
    g.add_nodes_from(exe_df.explode('neighbourhood').reset_index()["index"].unique())
    g.add_edges_from([edge for edge in exe_df.explode('neighbourhood').reset_index().dropna()[["index","neighbourhood"]].values.tolist() \
                      if edge[1] in exe_df.index.unique()])
    return nx.adjacency_matrix(g).toarray()

def generate_minhash(perms,lib_calls):
    m=MinHash(num_perm=perms)
    if lib_calls==[]:
        return list(np.full((perms,),np.inf))
    else:
        for lib_call in lib_calls:
            if "thunk" in lib_call or "FUN" in lib_call:
                continue
            m.update(lib_call.encode('utf8'))
        return list(m.digest())

def compute_fingerprint(exe_df,adj,node_embedding_detail):
    nodes_minhashes=np.array(exe_df.features.apply(lambda x:generate_minhash(node_embedding_detail,x)).to_list())
    neighbourhoods_minhashes=np.array([np.min(np.vstack((nodes_minhashes[i],nodes_minhashes[adj[i]==1])),axis=0) for i in range(adj.shape[0])])
    return np.sort(neighbourhoods_minhashes,axis=0)

def extract_fingerprint(exe_df,node_embedding_detail,neighbourhood_embedding):
    adj=compute_adj_mat(exe_df)
    fingerprint=compute_fingerprint(exe_df,adj,node_embedding_detail)
    if fingerprint.shape[0]<neighbourhood_embedding:
        fingerprint=np.vstack((fingerprint,np.full((neighbourhood_embedding-fingerprint.shape[0],node_embedding_detail),np.inf)))
    if fingerprint.shape[0]>neighbourhood_embedding:
        fingerprint=fingerprint[:neighbourhood_embedding]
    fingerprint[fingerprint==np.inf]=0
    return fingerprint

def main_extraction(target_folder=None, output_file=None):
    with open('config/config.json','r') as json_config:
            config=json.load(json_config)

    if target_folder is None:
        target_folder = config["feature_extractor_settings"]["raw_input_data_folder"]
    if output_file is None:
        output_file = f"{config['feature_extractor_settings']['output_directory']}{config['feature_extractor_settings']['output_name']}"

    families_db=pd.read_parquet("data/other/families_df.parquet")
    start=time.time()

    xlist = []
    for name in os.listdir(target_folder):
        if "_" in name:
            only_hash_name = name.split("_")[0]
        else:
            only_hash_name = name.replace(".json", "")
        try:
            if name != ".gitkeep":
                xlist += [[only_hash_name,\
                            families_db.loc[families_db.name==only_hash_name].category.values[0],\
                            extract_fingerprint(\
                                                pd.DataFrame(json.load(open(os.path.join(target_folder, name)))["nodes"]).transpose(),\
                                                config["feature_extractor_settings"]["node_embedding_detail"],\
                                                config["feature_extractor_settings"]["neighbourhood_embedding"]\
                                                )]]
        except KeyError:
            print(name)
    processed = pd.DataFrame(xlist, columns=["name","category","fingerprint"])
    end=time.time()
    print(f"{target_folder.split('/')[-1]} {timer(start,end)}")
    processed["shape"]=processed.fingerprint.apply(lambda x: list(x.shape))
    processed["fingerprint"]=processed.fingerprint.apply(lambda x: x.ravel())
    processed.to_parquet(output_file)


if __name__ == "__main__":
    
    exp_root = "/home/fra/Desktop/storage/out/meloni_samples_5/evasion_exps_new_bal/"
    df_outs = os.path.join(exp_root, "dfout")
    exp_root = os.path.join(exp_root, "data")
    if not os.path.exists(df_outs):
        os.mkdir(df_outs)
    all_exp_f = []
    for kind in os.listdir(exp_root):
        for amount in os.listdir(os.path.join(exp_root, kind)):
            all_exp_f += [os.path.join(exp_root, kind, amount)]
    print(all_exp_f)

    for graph_folder in all_exp_f:
        tic = time.time_ns()
        df_name = "{}.parquet".format(os.path.basename(graph_folder))
        df_path = os.path.join(df_outs, df_name)

        main_extraction(target_folder=graph_folder, output_file=df_path)
        toc = time.time_ns()
        print(f"done {df_name} in {toc - tic}")
    """
    tf = "/storage/out/meloni_samples_5/graphs_restricted_at_100xfam/"
    of = "/home/fra/Desktop/storage/out/meloni_samples_5/evasion_exps/out/df_outs/"
    df_name = "original.parquet"
    df_path = os.path.join(of, df_name)
    main_extraction(target_folder=tf, output_file=df_path)
    """
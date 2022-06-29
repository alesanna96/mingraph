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

with open('./config/config.json','r') as json_config:
        config=json.load(json_config)

families_db=pd.read_parquet(config["local_dbs"]["families_db"])
start=time.time()
"""
processed=pd.DataFrame([[name.replace(".json",""),\
                        families_db.loc[families_db.name==name.replace(".json","")].category.values[0],\
                        extract_fingerprint(\
                                            pd.DataFrame(json.load(open(f'{config["feature_extractor_settings"]["raw_input_data_folder"]}{name}'))["nodes"]).transpose(),\
                                            config["feature_extractor_settings"]["node_embedding_detail"],\
                                            config["feature_extractor_settings"]["neighbourhood_embedding"]\
                                            )] \
                        for name in os.listdir(config["feature_extractor_settings"]["raw_input_data_folder"]) \
                        if name!=".gitkeep"],\
                        columns=["name","category","fingerprint"])
"""

xlist = []
for name in os.listdir(config["feature_extractor_settings"]["raw_input_data_folder"]):
    try:
        if name != ".gitkeep":
            xlist += [[name.replace(".json",""),\
                        families_db.loc[families_db.name==name.replace(".json","")].category.values[0],\
                        extract_fingerprint(\
                                            pd.DataFrame(json.load(open(f'{config["feature_extractor_settings"]["raw_input_data_folder"]}{name}'))["nodes"]).transpose(),\
                                            config["feature_extractor_settings"]["node_embedding_detail"],\
                                            config["feature_extractor_settings"]["neighbourhood_embedding"]\
                                            )]]
    except KeyError:
        print(name)
processed = pd.DataFrame(xlist, columns=["name","category","fingerprint"])
end=time.time()
print(f"{config['feature_extractor_settings']['raw_input_data_folder'].split('/')[-2]} {timer(start,end)}")
processed["shape"]=processed.fingerprint.apply(lambda x: list(x.shape))
processed["fingerprint"]=processed.fingerprint.apply(lambda x: x.ravel())
processed.to_parquet(f"{config['feature_extractor_settings']['output_directory']}{config['feature_extractor_settings']['output_name']}")

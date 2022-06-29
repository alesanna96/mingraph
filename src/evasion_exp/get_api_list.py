import json
import os
import numpy as np

if __name__ == "__main__":
    rf = r"/storage/out/meloni_samples_5/graphs_balanced/"
    graph_list = [os.path.join(rf, y) for y in os.listdir(rf)]
    jsondocs = []
    for x in graph_list:
        with open(x, "r") as jf:
            jsondocs += [json.load(jf)]
    totfeat = []

    for jd in jsondocs:
        for fun in jd["nodes"]:
            totfeat += jd["nodes"][fun]["features"]
    
    totfeat = [x for x in totfeat if not x.startswith("FUN_")]
    dictfeats = dict()

    for i in totfeat:
        if i in dictfeats.keys():
            dictfeats[i] += 1
        else:
            dictfeats.update({i: 1})
    
    with open("dfeats_balanced.json", "w") as jf:
        json.dump(dictfeats, jf, indent=4)
        

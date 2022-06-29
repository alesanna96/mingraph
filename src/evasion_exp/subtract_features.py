import json
import os


if __name__ == "__main__":
    json_file = r"/home/fra/Documents/thesis/src/evasion_exp/sorted_dfeats_balanced.json"

    with open(json_file, "r") as fp:
        topn = json.load(fp)
    
    n = 200
    topn = list(topn.keys())[:n]

    jfolder = r"/storage/out/meloni_samples_5/graphs_balanced/"
    jlist = [os.path.join(jfolder, jname) for jname in os.listdir(jfolder)]
    jlist_len = len(jlist)

    for subtract_top_x in range(10, n + 1, 10):
        print("subtraction of top {} begun".format(subtract_top_x))
        for index, jpath in enumerate(jlist):
            of = "/home/fra/Desktop/storage/out/meloni_samples_5/evasion_exps_new_bal/subtract/top_{}_removed".format(subtract_top_x)
            if not os.path.exists(of):
                os.mkdir(of)
            name, ext = os.path.basename(jpath).split('.')
            destname = "{}_minus_top_{}.{}".format(name, subtract_top_x, ext)
            destpath = os.path.join(of, destname)

            with open(jpath, "r") as jr:
                jdata = json.load(jr)
            
            for fun in jdata["nodes"]:
                jdata["nodes"][fun]["features"] = \
                    [x for x in jdata["nodes"][fun]["features"] if x not in topn[:subtract_top_x]]
            
            with open(destpath, "w") as jw:
                json.dump(jdata, jw, indent=4)
            
            print("\r{}/{}".format(index, jlist_len), end="")
        print("\nsubtraction of top {} done".format(subtract_top_x))

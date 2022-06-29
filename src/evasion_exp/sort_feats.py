import json

if __name__ == "__main__":
    with open("src/evasion_exp/dfeats_balanced.json", "r") as jf:
        dfeats = json.load(jf)

    sorted_dfeats = dict(sorted(dfeats.items(), key=lambda x: x[1], reverse=True))

    with open("src/evasion_exp/sorted_dfeats_balanced.json", "w") as jf:
        json.dump(sorted_dfeats, jf, indent=4)
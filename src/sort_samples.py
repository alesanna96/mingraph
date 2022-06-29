import pandas as pd
import json
import os
import shutil


if __name__ == "__main__":
    with open('./config/config.json','r') as json_config:
        config=json.load(json_config)["sort_samples_settings"]
    
    output_folder = os.path.join(config["data_folder"], config["output_folder"])
    try:
        os.mkdir(output_folder)
    except Exception:
        pass
    
    unpacked_folder = os.path.join(output_folder, "unpacked")
    packed_folder = os.path.join(output_folder, "packed")
    try:
        os.mkdir(unpacked_folder)
        os.mkdir(packed_folder)
    except Exception:
        pass

    input_folder = os.path.join(config["data_folder"], config["raw_samples"])
    packing_detection_result = pd.read_parquet(config["input_parquet"])

    unpacked_files = packing_detection_result.name[packing_detection_result.packed == 0]
    packed_files = packing_detection_result.name[packing_detection_result.packed == 1]
    
    [shutil.copy2(os.path.join(input_folder, x), unpacked_folder) for x in unpacked_files]
    [shutil.copy2(os.path.join(input_folder, x), packed_folder) for x in packed_files]

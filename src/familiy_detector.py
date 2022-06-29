import pandas as pd 
import subprocess 
import os 
import json

with open('./config/config.json','r') as json_config:
        config=json.load(json_config)

jsons=os.listdir(config["family_detector_settings"]["VirusTotal_jsons_folder"])

df=pd.DataFrame({'name':[sample.replace('.json','') for sample in jsons],\
                'category':[subprocess.check_output([config["family_detector_settings"]["avclass_labeler_path"],\
                                                    '-vt',\
                                                    f'{config["family_detector_settings"]["VirusTotal_jsons_folder"]}{json}'])\
                            .split(b'\t')[1].strip().decode('ascii') \
                            for json in jsons]})

df.to_parquet(f'{config["family_detector_settings"]["output_directory"]}{config["family_detector_settings"]["output_name"]}')
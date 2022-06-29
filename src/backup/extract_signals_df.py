import pandas as pd 
import subprocess 
import os
from extract_signal import SignalExtractor 

jsons=os.listdir('./data/extracted_code/')
jsons.remove('.gitkeep')

def extract_signal(path):
    extr=SignalExtractor('./data/extracted_code/'+path)
    extr.generate_minhash()
    extr.generate_signal()
    return extr.signal

df=pd.DataFrame({'name':[sample.replace('.json','') for sample in jsons],\
                'category':[subprocess.check_output(['/home/fra/Documents/avclass/avclass_labeler.py','-vt',f'./data/vt_jsons/{json}']).split(b'\t')[1].strip().decode('ascii') for json in jsons],\
                'signal':[extract_signal(json) for json in jsons]\
                })
df['signal_length']=df['signal'].apply(lambda x:x.size)
df.to_parquet('./data/signals/signals_df.parquet')
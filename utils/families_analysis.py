import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_parquet('./data/other/families_df.parquet')
df2=df.loc[~df.category.str.contains('SINGLETON')]
df2.category.value_counts()[df2.category.value_counts()>=500].plot(kind='bar',fontsize=8)
plt.show()

with open('./data/other/samples_to_select.txt','w') as out:
    for name in df2.loc[df2.category.isin(df2.category.value_counts()[df2.category.value_counts()>=500].index)].name.to_list():
        out.write(name+'\n')
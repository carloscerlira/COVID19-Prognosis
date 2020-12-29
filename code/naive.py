import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.naive_bayes import GaussianNB



from sbs import SBS
from sfs import SFS

from gen_data import gen_data_cdmx, get_conf, predict
df_com, df_sin, df_hosp = gen_data_cdmx()
for df in (df_com, df_sin, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, GaussianNB, name="naive/cdmx_"+df.name+".txt")
    SBS(df, 1, GaussianNB,  name="naive/cdmx_sbs_"+df.name)
    SFS(df, df.shape[1]-2, GaussianNB, name="naive/cdmx_sfs_"+df.name)
    

from gen_data import gen_data_mx, predict  
df_com, df_hosp = gen_data_mx()
for df in (df_com, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, GaussianNB, name="naive/mx_"+df.name+".txt")
    SBS(df, 1, GaussianNB,  name="naive/mx_sbs_"+df.name)
    SFS(df, df.shape[1]-2, GaussianNB,  name="naive/mx_sfs_"+df.name)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# from sbs import SBS
# from sfs import SFS

from gen_data import gen_data_cdmx, get_conf, predict
df_com, df_sin, df_hosp = gen_data_cdmx()
for df in (df_com, df_sin, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, GaussianNB, name="naive/cdmx/"+df.name+".txt")

from gen_data import gen_data_mx, predict  
df_com, df_hosp = gen_data_mx()
for df in (df_com, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, GaussianNB, name="naive/mx/"+df.name+".txt")

# from sbs import SBS 
# from sfs import SFS
# SBS(data=df_hosp, q=1, clasificador=GaussianNB, ExitosPorDimension=[])
# SFS(data=df_hosp, q=1, clasificador=GaussianNB, ExitosPorDimension=[])

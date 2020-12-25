import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.neighbors import KNeighborsClassifier

from gen_data import gen_data_cdmx, get_conf, predict
df_com, df_sin, df_hosp = gen_data_cdmx()
for df in (df_com, df_sin, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, lambda: KNeighborsClassifier(n_neighbors=10), name="knn/cdmx_"+df.name+".txt")

from gen_data import gen_data_mx, predict  
df_com, df_hosp = gen_data_mx()
for df in (df_com, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, lambda: KNeighborsClassifier(n_neighbors=10), name="knn/mx_"+df.name+".txt")
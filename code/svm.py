import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.svm import SVC

from gen_data import gen_data_cdmx, get_conf, predict
df_com, df_sin, df_hosp = gen_data_cdmx()
for df in (df_com, df_sin, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, SVC, name="svm/cdmx/"+df.name+".txt")

from gen_data import gen_data_mx, predict  
df_com, df_hosp = gen_data_mx()
for df in (df_com, df_hosp):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    predict(X, y, SVC, name="svm/mx/"+df.name+".txt")
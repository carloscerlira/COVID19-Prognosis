import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")


def gen_data():
    url = "https://raw.githubusercontent.com/carloscerlira/Datasets/master/COVID19CDMX/train.csv"
    df = pd.read_csv(url);

    columns = ['fechreg', 'sexo', 'fecdef', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea','mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito','dolabdo', 'conjun', 	'cianosis',
            'inisubis','diabetes', 'epoc', 'asma', 'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin', 'tipacien']

    dfmini = df.loc[:,columns]
    dfmini = dfmini[dfmini['resdefin'].isin(['NEGATIVO','SARS-CoV-2'])] 

    dfmini["fecdef"].fillna("NO", inplace=True)
    dfmini["intubado"].fillna("NO", inplace=True)
    dfmini["estaemba"].fillna("NO", inplace=True)
    dfmini["digcline"].fillna("NO", inplace=True)
    dfmini["fecdef"] = dfmini["fecdef"].map(lambda x: "SI" if x != "NO" else x)

    cat = {}
    for col in dfmini:
        if col in ["fechreg", "edad"]: continue 
        lookup = {x:i for i,x in enumerate(dfmini[col].unique())}
        cat[col] = lookup
        dfmini[col] = dfmini[col].map(lookup)

    dfmini['edad']= (dfmini['edad']-dfmini['edad'].mean())/dfmini['edad'].std()

    col_com  = ['sexo', 'tipacien', 'edad', 'diabetes',	'epoc', 'asma',	'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin', 'fecdef']

    col_sin = ['sexo', 'tipacien', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea', 'mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito', 'dolabdo', 'conjun', 'cianosis',
            'inisubis', 'resdefin']

    df_com = dfmini.loc[:,col_com]
    df_com = df_com[df_com['resdefin']==1]
    df_sin = dfmini.loc[:,col_sin]

    df_hosp = dfmini.copy()
    df_hosp.drop('fechreg', axis=1, inplace=True)
    df_hosp.drop('fecdef', axis=1, inplace=True)
    df_hosp.drop('resdefin', axis=1, inplace=True)
    df_hosp.loc[df_hosp.intubado==1, 'tipacien'] = 1
    df_hosp.drop('intubado', axis=1, inplace=True)
    
    return df_com, df_sin, df_hosp

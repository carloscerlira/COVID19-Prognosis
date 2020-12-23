import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
plt.style.use("seaborn")

def gen_data_cdmx():
    url = "https://raw.githubusercontent.com/carloscerlira/Datasets/master/COVID19CDMX/train.csv"
    df = pd.read_csv(url);

    columns = ['tipacien', 'fechreg', 'sexo', 'fecdef', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea','mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito','dolabdo', 'conjun', 	'cianosis',
            'inisubis','diabetes', 'epoc', 'asma', 'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin']

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

    col_com  = ['fecdef', 'sexo', 'tipacien', 'edad', 'diabetes', 'epoc', 'asma', 'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin']

    col_sin = ['resdefin', 'sexo', 'tipacien', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea', 'mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito', 'dolabdo', 'conjun', 'cianosis',
            'inisubis']

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

def gen_data_mx():
    return df_com, df_hosp

def get_conf(X, y, classifier):
    conf = [[0,0],[0,0]]
    for clf in [0, 1]:
        for pred_clf in [0, 1]:
            X_clf = X[y == clf]
            y_clf = y[y == clf]
            y_pred = classifier.predict(X_clf)
            cnt = len(y_pred[y_pred == pred_clf])
            prob = cnt/len(y_clf)
            conf[clf][pred_clf] = prob
    TP, FP, FN, TN = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    fm = 2*TP/(2*TP+FP+FN)
    recall = TP/(TP+FN)
    print("Accuaracy: ", acc)
    print("Precision: ", prec)
    print("f-measure: ", fm)
    print("Recall: ", recall)
    return conf 

def predict(X, y, gen_clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_train = np.concatenate((X_train[y_train==0][:len(y_train[y_train==1])], X_train[y_train==1]))
    y_train = np.concatenate((y_train[y_train==0][:len(y_train[y_train==1])], y_train[y_train==1]))
    
    clf = gen_clf()
    y_pred = clf.fit(X_train, y_train)
    
    print(clf.score(X_test, y_test))
    conf = get_conf(X_test, y_test, clf)
    for row in conf:
        print(row)

import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use("seaborn")
from sklearn.model_selection import train_test_split

def gen_data_cdmx():
    #url = "https://raw.githubusercontent.com/carloscerlira/Datasets/master/COVID19CDMX/train.csv"
    # df = pd.read_csv(url)
    url = "C:/Users/i5 8400/Desktop/Ciencias de Datos/Datasets/COVID19CDMX/datos.csv"
    # url = "C:/Users/artem/Documents/Ciencia-Datos/BD-covid/datos.csv"
    df = pd.read_csv(url, sep=";")

    columns = ['tipacien', 'fechreg', 'sexo', 'fecdef', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea','mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito','dolabdo', 'conjun', 'cianosis',
            'inisubis','diabetes', 'epoc', 'asma', 'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin']

    dfmini = df.loc[:,columns]
    dfmini = dfmini[dfmini['resdefin'].isin(['NEGATIVO','SARS-CoV-2'])] 

    dfmini["fecdef"].fillna("NO", inplace=True)
    dfmini["intubado"].fillna("NO", inplace=True)
    dfmini["estaemba"].fillna("NO", inplace=True)
    dfmini["digcline"].fillna("NO", inplace=True)
    dfmini["fecdef"] = dfmini["fecdef"].map(lambda x: "SI" if x != "NO" else x)
    dfmini["edad"]= (dfmini["edad"]-dfmini["edad"].mean())/dfmini["edad"].std()

    cat = {}
    for col in dfmini:
        if col in ["fechreg", "resdefin", "edad"]: continue 
        lookup = {x:i for i,x in enumerate(dfmini[col].unique())}
        cat[col] = lookup
        dfmini[col] = dfmini[col].map(lookup)

    map_resdefin = {"NEGATIVO":0, "SARS-CoV-2":1}
    cat["resdefin"] = map_resdefin
    dfmini["resdefin"] = dfmini["resdefin"].map(map_resdefin)


    col_com = ['fecdef', 'sexo', 'edad', 'diabetes', 'epoc', 'asma', 'inmusupr', 'hiperten', 'vihsida', 'otracon', 'enfcardi', 'obesidad', 'insrencr',
            'tabaquis', 'resdefin']

    col_sin = ['resdefin', 'sexo', 'intubado', 'digcline', 'edad', 'estaemba', 'fiebre', 'tos', 'odinogia', 'disnea', 'irritabi', 'diarrea', 
            'dotoraci','calofrios', 'cefalea', 'mialgias', 'artral', 'ataedoge', 'rinorrea', 'polipnea', 'vomito', 'dolabdo', 'conjun', 'cianosis',
            'inisubis']

    df_com = dfmini.loc[:,col_com]
    df_com = df_com[df_com['resdefin']==1]
    df_com.drop('resdefin', axis=1, inplace=True)

    df_sin = dfmini.loc[:,col_sin]

    df_hosp = dfmini.copy()
    df_hosp.drop('fechreg', axis=1, inplace=True)
    df_hosp.drop('fecdef', axis=1, inplace=True)
    df_hosp.drop('resdefin', axis=1, inplace=True)
    # df_hosp.loc[df_hosp.intubado==1, 'tipacien'] = 1
    df_hosp.drop('intubado', axis=1, inplace=True)
    
    df_com.name = "com"
    df_sin.name = "sin"
    df_hosp.name = "hosp"
    return df_com, df_sin, df_hosp

def gen_data_mx():
    # url = "https://raw.githubusercontent.com/carloscerlira/Datasets/master/COVIDMX/train.csv"
    # df = pd.read_csv(url)
    url = "C:/Users/i5 8400/Desktop/Ciencias de Datos/Datasets/COVIDMX/datos.csv"
    # url = r"C:/Users/artem/Documents/Ciencia-Datos/BD-covid/datosmx.csv"
    df = pd.read_csv(url, encoding="latin")
    
    col_com  = ['FECHA_SINTOMAS', 'SEXO', 'TIPO_PACIENTE', 'FECHA_DEF', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO', 'DIABETES',	'EPOC', 'ASMA',	'INMUSUPR', 
           'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA','TABAQUISMO', 'CLASIFICACION_FINAL', 'UCI']
    df_mx =  df.loc[:,col_com].copy()

    map_yes_no = {1.0:1, 2.0:0, 97.0:0, 98.0:0, 99.0:0}
    map_clf_final = {1.0:1, 2.0:1, 3.0:1, 4.0:0, 5.0:0, 6.0:0, 7.0:0}
    
    df_mx["CLASIFICACION_FINAL"] = df_mx["CLASIFICACION_FINAL"].map(map_clf_final)
    df_mx["SEXO"] = df_mx["SEXO"]-1
    df_mx["TIPO_PACIENTE"] = df_mx["TIPO_PACIENTE"]-1
    df_mx["FECHA_DEF"] = df_mx["FECHA_DEF"].map(lambda x: 0 if x == "9999-99-99" else 1)
    df_mx["EDAD"]=(df_mx["EDAD"]-df_mx["EDAD"].mean())/df_mx["EDAD"].std()
    
    for col in df_mx:
        if col not in ["FECHA_SINTOMAS", "TIPO_PACIENTE", "FECHA_DEF", "CLASIFICACION_FINAL", "EDAD", "SEXO"]:
            df_mx[col] = df_mx[col].map(map_yes_no)

    df_com = df_mx[df_mx['CLASIFICACION_FINAL']==1]
    df_com.insert(1,'y',df_mx['FECHA_DEF'])
    df_com = df_com.drop('CLASIFICACION_FINAL', axis=1)
    df_com = df_com.drop('FECHA_DEF', axis=1)
    df_com = df_com.drop('FECHA_SINTOMAS', axis=1)
    df_com = df_com.drop('TIPO_PACIENTE', axis=1)


    df_hosp = df_mx[df_mx['CLASIFICACION_FINAL']==1]
    df_hosp.insert(1,'y',df_mx['TIPO_PACIENTE'])
    df_hosp = df_hosp.drop('TIPO_PACIENTE', axis=1)
    df_hosp = df_hosp.drop('CLASIFICACION_FINAL', axis=1)
    df_hosp = df_hosp.drop('FECHA_DEF', axis=1)
    df_hosp = df_hosp.drop('FECHA_SINTOMAS', axis=1)
    df_hosp = df_hosp.drop('UCI', axis=1)

    df_com.name = "com"
    df_hosp.name = "hosp"
    return df_com, df_hosp 

def get_conf(X, y, classifier): 
    cnt_P, cnt_N = len(y[y==1]), len(y[y==0])
    print(f"For test: {cnt_P/len(y): .3f}, {cnt_N/len(y): .3f}")
    conf = [[0,0],[0,0]]
    for clf in [0, 1]:
        for pred_clf in [0, 1]:
            X_clf = X[y == clf]
            y_clf = y[y == clf]
            y_pred = classifier.predict(X_clf)
            cnt = len(y_pred[y_pred == pred_clf])
            prob = cnt/len(y_clf)
            conf[clf][pred_clf] = prob
    TP, FP, FN, TN = conf[1][1]*cnt_P, conf[0][1]*cnt_N, conf[1][0]*cnt_P, conf[0][0]*cnt_N
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    fm = 2*TP/(2*TP+FP+FN)
    recall = TP/(TP+FN)
    print(f"Accuaracy: {acc: .3f}")
    print(f"Precision: {prec: .3f}")
    print(f"f-measure: {fm: .3f}")
    print(f"Recall: {recall: .3f}")
    print("Confussion Matrix: ")
    for row in conf:
        for x in row:
            print(f"{x:.3f}", end=" ")
        print()
    return conf 

def predict(X, y, gen_clf, name):
    orig_stdout = sys.stdout
    f = open(name, "w")
    sys.stdout = f
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # print("For test: ", len(y_test[y_test==0]), len(y_test[y_test==1]))
    X_train = np.concatenate((X_train[y_train==0][:len(y_train[y_train==1])], X_train[y_train==1]))
    y_train = np.concatenate((y_train[y_train==0][:len(y_train[y_train==1])], y_train[y_train==1]))
    # print("For train: ", len(y_train[y_train==0]), len(y_train[y_train==1]))
    clf = gen_clf()
    clf.fit(X_train, y_train)
    conf = get_conf(X_test, y_test, clf)
    
    sys.stdout = orig_stdout
    f.close()

# df_co, df_hosp = gen_data_mx()
# df_com, df_sin, df_hosp = gen_data_cdmx()
# print(df_sin.head())
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

def SBS(data,q,gen_clf, name,ExitosPorDimension = None):
  """
  Esta función aplica el algoritmo de Sequencial Backward Selection con k-vecinos

  Parámetros:
  - Entradas:
    - data = data set a analizar (asume que la primera columna tiene la clasificación)
    - q = número de dimensiones a seleccionar

  - Salidas:
    - ExitosPorDimension: Una lista de q elementos con la cantidad máxima de éxitos al eliminar una dimensión
    - peores: Lista en orden de las peores caracteristicas
  """
  # Este algoritmo recursivamente elimina la dimensión que al quitarla del análisis
  # hace que el número de éxitos de predicción sea máximo
  clf = gen_clf()
  global tamaño, peores
  if(not ExitosPorDimension):
        ExitosPorDimension = []
        tamaño = data.shape[1]-1
        peores = []
  #print(exitopordimension)
  if (q == tamaño):   # Si hemos llegado a la dimensión deseada
    # Graficamos
    x = np.arange(1,tamaño,1)
    plt.figure(figsize=(15,7))
    plt.plot(x,ExitosPorDimension, linewidth=2, color = "red")
    plt.xlabel('Número de dimensiones reducidas')
    plt.ylabel('Calificación al clasificar')
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.savefig(name+".pdf")
    plt.show()    
    f = open(name+".txt", "w")
    for exito in ExitosPorDimension:
        f.write(str(exito))
        f.write("\n")
    for caracteristica in peores:
        f.write(str(caracteristica))
        f.write("\n")
    f.close()
    return ExitosPorDimension, peores # Regresa la lista de éxitos por dimensión reducida
  else:
    exitos = np.zeros(data.shape[1]-1)  # Crea un contador de éxitos por cada columna
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    for j in range(X.shape[1]-1):
      datos = X.drop(X.columns[j], inplace=False, axis=1)
      X_train, X_test, y_train, y_test = train_test_split(datos, y, test_size=0.2, random_state=42)
      # Homogeneizamos los datos con los que hacer fit
      X_train = np.concatenate((X_train[y_train==0][:len(y_train[y_train==1])], X_train[y_train==1]))
      y_train = np.concatenate((y_train[y_train==0][:len(y_train[y_train==1])], y_train[y_train==1]))
      clf.fit(X_train,y_train)
      # Analizamos los score para cada dimension reducida
      exitos[j] = clf.score(X_test, y_test)  # Aplica el clasificador de K vecinos cercanos
    # Tomamos el mayor score
    print('La mayor cantidad de éxitos es: ', max(exitos))
    ExitosPorDimension.append(max(exitos))
    peor_caracteristica = np.where(exitos == max(exitos))[0][0]
    print('La dimensión reducida es: ' , X.columns[peor_caracteristica])
    # Guardamos las peores caracteristicas en orden
    peores.append(X.columns[peor_caracteristica])
    # Seleccionamos las mejores caracteristicas
    bestdata = X.drop(X.columns[peor_caracteristica], inplace=False, axis=1)  # Quita la columna que menos ayuda
    bestdata.insert(0,'y', y)
    SBS(bestdata,q+1,gen_clf,name,ExitosPorDimension) # Llamada recursiva


    return ExitosPorDimension, peores

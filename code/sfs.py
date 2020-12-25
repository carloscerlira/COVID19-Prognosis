def SFS(data,q,clasificador,ExitosPorDimension=[]):
  """
  Esta función aplica el algoritmo de Sequencial Forward Selection con un clasificador dado

  Parámetros:
  - Entradas:
    - data = data set a analizar  (Asume que la primera columna es la categoría a usar)
    - q = número de dimensiones a seleccionar
    - Clasificador = El clasificador a usar

  - Salidas:
    - ExitosPorDimension: Una lista de q elementos con la cantidad máxima de éxitos al agregar una dimensión
    - mejores: Lista en orden de las mejores caracteristicas
  """
  # Este algoritmo recursivamente agrega una dimensión que al agregarla al análisis
  # hace que el número de éxitos de predicción sea máximo

  global bestdata, tamaño, mejores
  if(not ExitosPorDimension):
        tamaño = data.shape[1]-1
        bestdata = pd.DataFrame(data.y)
        mejores = []
  if (q == 0):   # Si hemos llegado a la dimensión deseada
    # Grafica de resultados
    plt.figure(figsize=(15,7))
    plt.plot(ExitosPorDimension, linewidth=2, color = "red")
    plt.xlabel('Número de dimensiones reducidas')
    plt.ylabel('Calificación al clasificar')
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.show()
    return ExitosPorDimension, mejores # Regresa la lista de éxitos por dimensión reducida
  else:
    exitos = np.zeros(data.shape[1]-1)  # Crea un contador de éxitos por cada columna
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    for j in range(X.shape[1]-1):
      datos = bestdata.copy()
      # Agregamos las caracteristicas una a una a best data para ver cual es la siguiente mejor
      datos.insert(datos.shape[1],X.columns[j], X.iloc[:,j])
      X_train, X_test, y_train, y_test = train_test_split(datos.iloc[:,1:], datos.iloc[:,0], test_size=0.4, random_state=42)
      # Homogeneizamos los datos con los que hacer fit
      X_train = np.concatenate((X_train[y_train==0][:len(y_train[y_train==1])], X_train[y_train==1]))
      y_train = np.concatenate((y_train[y_train==0][:len(y_train[y_train==1])], y_train[y_train==1]))
      clasificador.fit(X_train,y_train)
      # Analizamos los score para cada dimension reducida
      exitos[j] = clasificador.score(X_test, y_test)  # Aplica el clasificador de K vecinos cercanos
    # Tomamos el mayor score
    print('La mayor cantidad de éxitos es: ', max(exitos))
    ExitosPorDimension.append(max(exitos))
    mejor_caracteristica = np.where(exitos == max(exitos))[0][0]
    print('La caracteristica seleccionada es: ' , X.columns[mejor_caracteristica])
    # Guardamos las mejores caracteristicas en orden
    mejores.append(X.columns[mejor_caracteristica])
    # Seleccionamos las mejores caracteristicas
    bestdata.insert(bestdata.shape[1],X.columns[mejor_caracteristica],X.iloc[:,mejor_caracteristica])  # Agrega a bestdata la mejor caracteristica
    # Quitamos la caracteristica ya considerada
    data.drop(X.columns[mejor_caracteristica], inplace = True, axis=1)
    SFS(data,q-1, clasificador, ExitosPorDimension) # Llamada recursiva
    return ExitosPorDimension, mejores
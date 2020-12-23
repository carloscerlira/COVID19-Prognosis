def SBS(data,q,clasificador):
  """
  Esta función aplica el algoritmo de Sequencial Backward Selection con k-vecinos

  Parámetros:
  - Entradas:
    - data = data set a analizar
    - q = número de dimensiones a seleccionar

  - Salidas:
    - ExitosPorDimension: Una lista de q elementos con la cantidad máxima de éxitos al eliminar una dimensión
  """
  # Este algoritmo recursivamente elimina la dimensión que al quitarla del análisis
  # hace que el número de éxitos de predicción sea máximo

  global tamaño, ExitosPorDimension
  if(not ExitosPorDimension):
        ExitosPorDimension = []
        tamaño = data.shape[1]-1
        print(tamaño)
  #print(exitopordimension)
  if (q == tamaño):   # Si hemos llegado a la dimensión deseada
    return (ExitosPorDimension) # Regresa la lista de éxitos por dimensión reducida
  else:
    exitos = np.zeros(data.shape[1]-1)  # Crea un contador de éxitos por cada columna
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    for j in range(X.shape[1]-1):
      datos = X.drop(X.columns[j], inplace=False, axis=1)
      X_train, X_test, y_train, y_test = train_test_split(datos, y, test_size=0.2, random_state=42)
      #print('analizando caracteristica ', j-1, ' de ', X_test.shape[1])
      #print(y_train)
      clasificador.fit(X_train,y_train)
      exitos[j] = clasificador.score(X_test, y_test)  # Aplica el clasificador de K vecinos cercanos
    #print('Los exitos son: ', exitos)
    print('La mayor cantidad de éxitos es: ', max(exitos))
    ExitosPorDimension.append(max(exitos))
    peor_caracteristica = np.where(exitos == max(exitos))[0][0]
    print('La dimensión reducida es: ' , X.columns[peor_caracteristica])
    #print(data.columns[peor_caracteristica+1])
    bestdata = X.drop(X.columns[peor_caracteristica], inplace=False, axis=1)  # Quita la columna que menos ayuda
    bestdata.insert(0,'y', y)
    SBS(bestdata,q+1) # Llamada recursiva
    return ExitosPorDimension


ExitosPorDimension = []
resultadosSBS = SBS(df_hosp.sample(frac=0.01, random_state=1),1)
plt.figure(figsize=(15,7))
plt.plot(resultadosSBS, linewidth=2, color = "red")
plt.xlabel('Número de dimensiones reducidas')
plt.ylabel('Calificación al clasificar')
plt.grid(color='gray', linestyle='-', linewidth=1)
plt.show()
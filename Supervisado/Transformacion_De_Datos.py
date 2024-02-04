
# Librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Lectura de datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/Datos.csv'
df = pd.read_csv(url)
print(f'Dataset: \n{df}')

datos = df['likes']
datos = datos.astype(float).fillna(0.0)
print(f'Datos (Columna - likes): \n{datos}')

# Converir los datos a una matriz

matriz = np.array(datos).reshape(-1, 1)
print(f'\nDatos convertidos a matriz: \n{matriz}')

# Escaladores

escalador_1 = preprocessing.RobustScaler()
escalador_2 = preprocessing.StandardScaler()

# Entrenamiento

columna = escalador_1.fit_transform(matriz)
df_1 = pd.DataFrame(columna)

columna = escalador_2.fit_transform(matriz)
df_2 = pd.DataFrame(columna)

# Poner nombres a las columnas

df_1.columns = ['RobustScaler']
df_2.columns = ['StandardScaler']

# Mostrar df
print(f'\nDF 1: \n{df_1}')
print(f'\nDF 2: \n{df_2}')

# Mostra datos de los df

print(f'\nCantidad de la matriz: {matriz.shape}')
print(f'\nDescripcion DF 1: \n{df_1.describe()}')
print(f'\nDescripcion DF 2: \n{df_2.describe()}')

# Grafica

datos.plot.kde(legend = True)
ejes = df_1.plot.kde(legend = True)
df_2.plot.kde(legend = True, ax = ejes)

plt.show()
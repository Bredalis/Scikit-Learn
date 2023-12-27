
# Librerias

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Lectura de datos

dataset = load_wine()

# Division de datos

x_train, x_test, y_train, y_test = train_test_split(
	dataset.data, dataset.target)

# Mostrar propiedades de datos

print(f'Dataset: \n{dataset}')
print(f'\nDatos: \n{dataset.data}')
print(f'\nEtiquetas: \n{dataset.target}')

print(f'\nCaracteristicas: \n{dataset.feature_names}')
print(f'Nombres de etiquetas: {dataset.target_names}')
print('Cantidad de clases de etiquetas:', 
	len(np.unique(dataset.target_names)))

print(f'\nX train: \n{x_train}')
print(f'\nX train cantidad: \n{x_train.shape}')
print(f'\nX test: \n{x_test}')
print(f'\nX test cantidad: \n{x_test.shape}')

print(f'\nY train: \n{y_train}')
print(f'\nY train cantidad: \n{y_train.shape}')
print(f'\nY test: \n{y_test}')
print(f'\nY test cantidad: \n{y_test.shape}')
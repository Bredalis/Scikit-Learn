
# Librerias

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Lectura de datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/Wine.csv'
df = pd.read_csv(url)
df = df.astype(float).fillna(0.0)

print(f'DF: \n{df}')
print(type(df))

# Division de datos

x = df.quality
y = df.quality

x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size = 0.2, random_state = 42)

# Mostrar datos

print(f'x train: \n {x_train}')
print(f'\nx test: \n {x_test}')
print(f'\ny train: \n {y_train}')
print(f'\ny test: \n {y_test}')

# Mostar cantidad de datos

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# Escalador 

escalador = StandardScaler()

x_train_array = escalador.fit_transform(x_train.values.reshape(-1, 1))
x_train = pd.DataFrame(x_train_array, index = x_train.index)

x_test_array = escalador.transform(x_test.values.reshape(-1, 1))
x_test = pd.DataFrame(x_test_array, index = x_test.index)

# Cambiar el nombre de la columna

x_train.columns = ['StandardScaler']
x_test.columns = ['StandardScaler']

# Mostrar df

print(f'\nDF x train: \n{x_train}')
print(f'\nDF x test: \n{x_test}')

# Modelo

clf = SVC(kernel = 'poly')

# Entrenamiento y prediccion

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(f'\nPrediccion: \n{y_pred}')
print(f'\nRedimiento: {clf.score(x_test, y_test)}')
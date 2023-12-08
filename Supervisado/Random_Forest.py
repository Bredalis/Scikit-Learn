
# Librerias

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Lectura de datos

df = pd.read_excel("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Datos_Regresion_Lineal.xlsx")
print(f"DF: \n{df}")

# Division de datos

x = df[["Reduccion de Solidos"]]
y = df[["Reduccion de la demanda de oxigeno"]]

print(f"X: {x}")
print(f"X cantidad: {x.shape}")
print(f"Y: {y}")
print(f"Y cantidad: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

print(f"X train: \n{x_train}")
print(f"X train cantidad: \n{x_train.shape}")
print(f"X test: \n{x_test}")
print(f"X test cantidad: \n{x_test.shape}")

print(f"Y train: \n{y_train}")
print(f"Y train cantidad: \n{y_train.shape}")
print(f"Y test: \n{y_test}")
print(f"Y test cantidad: \n{y_test.shape}")

# Modelo

clf = RandomForestRegressor()

# Entrenamiento y prediccion

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(f"Prediccion: \n{y_pred}")
print(f"Mean Absolute Error (Metrica): \n{mean_absolute_error(y_test, y_pred)}")
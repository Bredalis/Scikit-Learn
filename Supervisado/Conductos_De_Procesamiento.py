
# Librerias

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Lectura de datos

df_train = pd.read_csv("C:\\Users\\angelica Gerrero\\Desktop\\LenguajesDeProgramacion\\Datasets\\CSV\\california_housing_train.csv")
df_test = pd.read_csv("C:\\Users\\angelica Gerrero\\Desktop\\LenguajesDeProgramacion\\Datasets\\CSV\\california_housing_test.csv")

# Division de datos

x_train, y_train = df_train.to_numpy()[:, :-1], df_train.to_numpy()[:, -1]
x_test, y_test = df_test.to_numpy()[:, :-1], df_test.to_numpy()[:, -1]

print(f"DF Train: \n {df_train.head()}")
print(f"DF Test: \n {df_test.head()}")

print(f"Cantidad de x train: {x_train.shape}")
print(f"Cantidad de y train: {y_train.shape}")
print(f"Cantidad de x test: {x_test.shape}")
print(f"Cantidad de y test: {y_test.shape}")

# Modelo y Entrenamiento

STD_Scaler = StandardScaler().fit(x_train[:, :2])
Min_Max_Scaler = MinMaxScaler().fit(x_train[:, 2:])

def Procesador(x):

	a = np.copy(x)

	a[:, :2] = STD_Scaler.transform(x[:, :2])
	a[:, 2:] = Min_Max_Scaler.transform(x[:, 2:])

	return a

transformacion_procesamiento = FunctionTransformer(Procesador)

p1 = Pipeline([
	('Scaler', transformacion_procesamiento), ('Linear Regression', LinearRegression())
])

p2 = Pipeline([
	('Scaler', transformacion_procesamiento), ('KNN Regression', KNR(n_neighbors = 7))
])

p3 = Pipeline([
	('Scaler', transformacion_procesamiento), ('Random Forest', RFR(n_estimators = 10, max_depth = 7))
])

def Entrenamiento(modelo, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test):

	modelo.fit(x_train, y_train)

	train_pred = modelo.predict(x_train)
	test_pred = modelo.predict(x_test)

	print("Training error: " + str(mean_absolute_error(train_pred, y_train)))
	print("Error de entrenamiento: " + str(mean_absolute_error(train_pred, y_train)))

Entrenamiento(p1)

print(Procesador(x_test))
print(transformacion_procesamiento)

print(f"p1: {p1}")
print(f"p2: {p2}")
print(f"p3: {p3}")
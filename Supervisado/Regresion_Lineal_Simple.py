
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Lectura de datos

df = pd.read_excel("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Datos_Regresion_Lineal.xlsx")
print(f"DF: \n{df}")

# Division de datos

x = df[["Reduccion de Solidos"]]
y = df[["Reduccion de la demanda de oxigeno"]]

# Modelo

clf = LinearRegression()

# Entrenamiento

clf.fit(x, y)

# Grafica

plt.scatter(x, y)
plt.plot(x, clf.predict(x))

plt.title("Regresion Lineal Simple")
plt.legend(["Y", "Predicciones"])
plt.xlabel("Reduccion de Solidos")
plt.ylabel("Reduccion de la demanda de oxigeno")
plt.grid()
plt.show()
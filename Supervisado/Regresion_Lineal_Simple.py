
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Lectura de datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/casas.csv'
df = pd.read_csv(url)
print(f'DF: \n{df}')

# Division de datos

x = df[['A']]
y = df[['B']]

# Modelo

clf = LinearRegression()

# Entrenamiento

clf.fit(x, y)

# Grafica

plt.scatter(x, y)
plt.plot(x, clf.predict(x))

plt.title('Regresion Lineal Simple')
plt.legend(['Y', 'Predicciones'])
plt.xlabel('A')
plt.ylabel('B')
plt.grid()
plt.show()
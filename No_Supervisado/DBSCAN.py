
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Datos

url = 'C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/casas.csv'
df = pd.read_csv(url)

print(f'df: \n {df}')

# Modelo

modelo = DBSCAN(eps = 2, min_samples = 10)

# Entrenamiento

modelo.fit_predict(df)

# Grafica

plt.figure(figsize = (7.5, 7.5))

plt.scatter(df['A'], df['B'], c = df['A'])

plt.ylabel('House Price in Pesos (1:100,000)')
plt.xlabel('Years of Building Age')
plt.box(False)
plt.show()
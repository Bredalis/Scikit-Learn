
# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Datos

df = pd.read_excel("C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/Datos_Regresion_Lineal.xlsx")

x = df.drop(['Reduccion de Solidos'], axis = 1)
x_normal = (x - x.min()) / (x.max() - x.min())

print(f"Info: \n {df.info()}")
print(f"df: \n {df.head()}")
print(f"x: \n {x}")
print(f"Normal Features: \n {x_normal}")

valores = []

for i in range(1, 11):

	modelo = KMeans(n_clusters = i, max_iter = 300)
	modelo.fit(x_normal)

	valores.append(modelo.inertia_)

# Modelo

modelo = KMeans(n_clusters = 3, max_iter = 300)

# Entrenamiento

modelo.fit(x_normal)

x_normal["KMeans_Clusters"] = modelo.labels_

descomposicion = PCA(n_components = 2)
descomposicion_x = descomposicion.fit_transform(x_normal)

df_descomposicion = pd.DataFrame(data = descomposicion_x, columns = ["Componente_1", "Componente_2"])

concatenacion = pd.concat([df_descomposicion, x_normal[["KMeans_Clusters"]]], axis = 1)

# Grafica

plt.plot(range(1, 11), valores)
plt.xlabel("Numbers of groups")
plt.ylabel("WCSS")

fig = plt.figure(figsize = (6,  6))

axis = fig.add_subplot(1, 1, 1)

axis.set_title("Principal Components", fontsize = 20)
axis.set_ylabel("Component 1", fontsize = 15)
axis.set_xlabel("Component 2", fontsize = 15)

colores = np.array(["skyblue", "pink", "red"])

axis.scatter(
	x = concatenacion.Componente_1, y = concatenacion.Componente_2,
	c = colores[concatenacion.KMeans_Clusters], s = 50
)

plt.show()
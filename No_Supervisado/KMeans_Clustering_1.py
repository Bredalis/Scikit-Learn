
# Librerias

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Datos

df = make_blobs(
	n_samples = 200, centers = 4,
	n_features = 2, cluster_std = 1.6,
	random_state = 50
)

puntos = df[0]

print(f'Dataset: \n{df}')
print(f'Puntos: \n{puntos}')

# Modelo

modelo = KMeans(n_clusters = 4)

# Entrenamiento

modelo.fit(puntos)

grupos = modelo.cluster_centers_
y_km = modelo.fit_predict(puntos)

print(f'Grupo: \n{grupos}')
print(f'Etiquetas: \n{y_km}')

# Grafica

def modelo():

	fig, axis = plt.subplots()

	axis.scatter(df[0][:, 0], df[0][:, 1])
	plt.show()

modelo()

plt.scatter(puntos[y_km == 0, 0], 
	puntos[y_km == 0, 1], s = 50, color = 'green')

plt.scatter(puntos[y_km == 1, 0], 
	puntos[y_km == 1, 1], s = 50, color = 'blue')

plt.scatter(puntos[y_km == 2, 0], 
	puntos[y_km == 2, 1], s = 50, color = 'yellow')

plt.scatter(puntos[y_km == 3, 0], 
	puntos[y_km == 3, 1], s = 50, color = 'red')

plt.scatter(grupos[0][0], 
	grupos[0][1], marker = '*', s = 200, color = 'black')

plt.scatter(grupos[1][0], 
	grupos[1][1], marker = '*', s = 200, color = 'black')

plt.scatter(grupos[2][0], 
	grupos[2][1], marker = '*', s = 200, color = 'black')

plt.scatter(grupos[3][0], 
	grupos[3][1], marker = '*', s = 200, color = 'black')

plt.show()
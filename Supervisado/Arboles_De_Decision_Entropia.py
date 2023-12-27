
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from math import log
from sklearn import tree
from sklearn.model_selection import train_test_split

# Lectura de datos

url = 'C:/Users/Angelica Gerrero/Desktop/LenguajesDeProgramacion/Datasets/CSV/dataset.csv'
datos = pd.read_csv(url)

diabetes = datos[datos['diabetes'] == 0]
anemia = datos[datos['anaemia'] == 1]

edades = pd.Series([40, 30, 20, 50])
sexo = pd.Series([0, 1, 1, 0])

x_train, x_test, y_train, y_test = train_test_split(
	datos[['age', 'sex']], datos[['diabetes', 'anaemia']], test_size = 0.30
)

print(f'Logaritmo de base 2: {log(8, 2)}')
print(f'Entropia: {entropy([1 / 2, 1 / 2], base = 2)}')
print(f'Entropia: {entropy([10 / 10, 0 / 10], base = 2)}')

print(f'{edades.value_counts() / edades.size}')
print(f'{sexo.value_counts() / sexo.size}')
print(f'{entropy(edades.value_counts() / edades.size, base = 2)}')
print(f'{entropy(sexo.value_counts() / sexo.size, base = 2)}')

# Modelo

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)

# Entrenamiento

clf.fit(x_train, y_train)

print(f'Rendimiento: {clf.score(x_test, y_test)}')
print('Arbol: \n', tree.export_text(clf, feature_names = ['age', 'sex']))
print(f'Predicccion: {clf.predict([[70, 1]])}')
   
# Grafica

tree.plot_tree(clf, feature_names = ['age', 'sex'])

plt.figure(figsize = (6, 6))

plt.xlabel('Edad', fontsize = 20.0)
plt.ylabel('sexo', fontsize = 20.0)

plt.scatter(diabetes['age'], diabetes['sex'], label = 'diabetes')
plt.scatter(anemia['age'], anemia['sex'], label = 'anemia', 
	marker = '*', c = 'lightcoral', s = 200
)

plt.legend(bbox_to_anchor = (1, 0.15))
plt.show()
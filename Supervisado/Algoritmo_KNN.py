
# Librerias

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Lectura de datos

x, y = make_classification(n_samples = 200)

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Modelo

clf = KNeighborsClassifier()

# Entrenamiento

clf.fit(x_train, y_train)

# Prediccion

y_pred = clf.predict(x_test)

print(f'y_pred: \n {y_pred}')
print(f'Performance: {clf.score(x_test, y_test)}')
print(f'Confusion Matrix: \n {confusion_matrix(y_pred, y_test)}')

# Grafica

plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train)
plt.scatter(x_test[1, 1], x_test[1, 1], s = 100)
plt.grid()

plt.show()
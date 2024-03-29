
# Librerias

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Lectura de datos

x, y = make_classification(n_samples = 200)

X_train, X_test, y_train, y_test = train_test_split(x, y)

# Modelo

clf = KNeighborsClassifier()

# Entrenamiento

clf.fit(X_train, y_train)

# Prediccion

y_pred = clf.predict(X_test)

print(f'y_pred: \n {y_pred}')
print(f'Performance: {clf.score(X_test, y_test)}')
print(f'Confusion Matrix: \n {confusion_matrix(y_pred, y_test)}')

# Grafica

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.scatter(X_test[1, 1], X_test[1, 1], s = 100)
plt.grid()

plt.show()
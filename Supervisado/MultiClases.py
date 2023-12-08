
# Librerias

from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
  recall_score, f1_score, classification_report)

# Lectura de datos

dataset = load_wine()
print(f"Dataset:\n {dataset}")
print(f"Cantidad: {dataset.data.shape}")

# Division de datos

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target)

print(f"X train: \n{x_train}")
print(f"X train cantidad: \n{x_train.shape}")
print(f"X test: \n{x_test}")
print(f"X test cantidad: \n{x_test.shape}")

print(f"Y train: \n{y_train}")
print(f"Y train cantidad: \n{y_train.shape}")
print(f"Y test: \n{y_test}")
print(f"Y test cantidad: \n{y_test.shape}")

# Modelo

clf = MLPClassifier()

# Entrenamiento y prediccion

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# Uso de metricas

print(f"\nPrediccion: \n{y_pred}")
print(f"\nMatriz de confucion: {confusion_matrix(y_pred, y_test)}")
print(f"\nExactitud: {accuracy_score(y_pred, y_test)}")
print(f"\nRendimiento: \n{clf.score(x_test, y_test)}")
print("\nPrecision: \n", precision_score(y_test, y_pred, average = "macro"))
print(f1_score(y_test, y_pred, average = "macro"))
print(f"\nInforme: \n{classification_report(y_pred, y_test)}")
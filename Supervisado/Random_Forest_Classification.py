
# Librerias

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, 
    accuracy_score, classification_report)

# Lectura de datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/heart_Disease.csv'
df = pd.read_csv(url)

print(f'DF: \n{df}')
print(df['target'].value_counts())

# Divison de datos

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f'X: \n{x}')
print(f'Y: \n{y}')

print(f'X cantidad: \n{x.shape}')
print(f'Y cantidad: \n{y.shape}')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 99)

print(f'X Train: \n{x_train}')
print(f'X Train cantidad: \n{x_train.shape}')
print(f'X Test: \n{x_test}')
print(f'X Test cantidad: \n{x_test.shape}')

print(f'Y Train: \n{y_train}')
print(f'Y Train cantidad: \n{y_train.shape}')
print(f'Y Test: \n{y_test}')
print(f'Y Test cantidad: \n{y_test.shape}')

# Modelo

clf = RandomForestClassifier(
    criterion = 'gini', max_depth = 8,
    min_samples_split = 10, random_state = 5
)

# Entrenamiento y prediccion

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(f'Prediccion: \n {y_pred}')

print(f'Columnas: \n{df.columns}')
print(f'Caracteristicas Importantes: \n {clf.feature_importances_}')
print(np.argsort(clf.feature_importances_))

# Metricas

print(f'Matriz de confucion: \n{confusion_matrix(y_pred, y_test)}')
print(f'Exactitud: {accuracy_score(y_pred, y_test)}')
print(f'Reporte: \n{classification_report(y_test, y_pred)}')
print('Validacion cruzada: \n', 
    cross_val_score(clf, x_train, y_train, cv = 10))

# Grafica

sns.countplot(df['target'])

plt.title('Gráfico de conteo de variables objetivo')
plt.ylabel('Conteo de etiquetas')
plt.xlabel('Etiqueta')
plt.grid()
plt.show()
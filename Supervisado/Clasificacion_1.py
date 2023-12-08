
# Librerias

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Lectura de datos

iris = datasets.load_iris()

x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)

# Modelo

clf = LogisticRegression(solver = "lbfgs", random_state = 42)

# Entrenamiento

clf.fit(x, y)

x_2 = [[1.7], [1.5]]
y_2 = clf.predict(x_2)

# Grafica

x_2 = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = clf.predict_proba(x_2)
frontera_de_decision = x_2[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize = (8, 3))

plt.plot(x[y == 0], y[y == 0], "bs")
plt.plot(x[y == 1], y[y == 1], "g^")

plt.plot([frontera_de_decision, frontera_de_decision], [-1, 2], "k:", linewidth = 2)
plt.plot(x_2, y_proba[:, 1], "g-", linewidth = 2, label = "iris virginica")
plt.plot(x_2, y_proba[:, 0], "b--", linewidth = 2, label = "Not iris virginica")

plt.text(frontera_de_decision + 0.02, 0.15, "Decision  boundary", fontsize = 14, color = "k", ha = "center")

plt.xlabel("Petal width (cm)", fontsize = 14)
plt.ylabel("Probability", fontsize = 14)

plt.legend(loc = "center left", fontsize = 14)
plt.axis([0, 3, -0.02, 1.02])

plt.show()
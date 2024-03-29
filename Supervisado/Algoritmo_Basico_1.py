
# Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/country_stats.csv'
data = pd.read_csv(url, index_col = 'Country')

x = np.c_[data['GDP_per_capita']]
y = np.c_[data['Life_satisfaction']]

print(f'data: {data}')
print(f'x: {x}')
print(f'y: {y}')

# Modelo

model = linear_model.LinearRegression()

# Entrenamiento

model.fit(x, y)

intercept, slope = model.intercept_[0], model.coef_[0][0]
money = 35000
satisfaction = model.predict([[money]])[0][0]

print(f'Parameters: {intercept, slope}')
print(f'Happy contries: {satisfaction}')

# Grafica

data.plot(
	kind = 'scatter', x = 'GDP_per_capita', 
	y = 'Life_satisfaction', figsize = (5, 3))

x = np.linspace(0, 60000, 10000)

plt.plot(x, intercept + slope * x, 'b')
plt.plot([money, money], [0, satisfaction],  'r--')
plt.plot(money, satisfaction, 'ro')

plt.text(50000, 3.1, r'$b = 4.85$', fontsize = 14, color = 'b')
plt.text(50000, 2.2, r'$w = 4.91 \times 10^{-5}', fontsize = 14, color = 'b')
plt.text(25000, 5.0, r'Prediction = 5.96', fontsize = 14, color = 'b')

plt.axis([0, 60000, 0, 10])
plt.ylabel('Life Satisfaction')
plt.xlabel('GPD per caipta')

plt.show()

# Librerias

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Lectura de datos

url = 'C:/Users/Bradalis/Desktop/LenguajesDeProgramacion/Datasets/CSV/'
data = pd.read_csv(url + 'housing.csv')

# Division de datos

train, test = train_test_split(data, test_size = 0.2)

# Guardar datos

train.to_csv(url + 'train.csv', index = False)
test.to_csv(url + 'test.csv', index = False)

train_data, y_train = train.drop(
    ['median_house_value'], axis = 1), train['median_house_value'].copy()

test_data, y_test = test.drop(
    ['median_house_value'], axis = 1), test['median_house_value'].copy()

train_numeric = train_data.drop(['ocean_proximity'], axis = 1)
train_categorical = train_data[['ocean_proximity']]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
])

num_attributes = list(train_numeric)
cat_attributes = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes)
])

X_train = full_pipeline.fit_transform(train_data)
X_test = full_pipeline.transform(test_data)

print(f'Data: \n{data.head()}')
print(f'Info: \n{data.info()}')

print(f'Data Length: {len(data)}')
print(f'Training Data Length: {len(train)}')
print(f'Test Data Length: {len(test)}')
print(f'Training Data: {X_train}')
print(f'Test Data: {X_test}')

# Modelos

model = LinearRegression()
model_2 = DecisionTreeRegressor()

# Entrenamiento

model.fit(X_train, y_train)
model_2.fit(X_train, y_train)

# Predicciones

y_pred = model.predict(X_test)
y_pred_2 = model_2.predict(X_test)

imputer = SimpleImputer(strategy = 'median')
imputer.fit(train_numeric)

X_train_numeric = imputer.transform(train_numeric)

scaler = StandardScaler()
scaler.fit(X_train_numeric)

X_train_numeric_scaler = scaler.transform(X_train_numeric)

categorical = OneHotEncoder()
X_train_categorical = categorical.fit_transform(train_categorical)
X_train_categorical.toarray()

print(f'Calculated values: {imputer.statistics_}')
print(f'Median: {X_train_numeric}')
print(f'Mean and statistics: {X_train_numeric_scaler}')
print(X_train_categorical)
print(f'Model 1 Prediction: {y_pred}')
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(f'Model 2 Prediction: {y_pred_2}')
print(np.sqrt(mean_squared_error(y_test, y_pred_2)))
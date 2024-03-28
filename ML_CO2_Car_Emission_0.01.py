#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:06:53 2024

@author: jedson
"""
# importar pacotes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

base = pd.read_csv('CO2 Emissions.csv')

X_fuel = pd.DataFrame(base['Fuel Type'])
X_transmission = pd.DataFrame(base['Transmission'])
X_consumption = pd.DataFrame(base[['Engine Size(L)','Cylinders','Fuel Consumption City (L/100 km)',
                       'Fuel Consumption Hwy (L/100 km)']])
y = pd.DataFrame(base['CO2 Emissions(g/km)'])
#y = base.iloc[:,11].values

X_consumption[X_consumption['Engine Size(L)'].isna()].count()
X_fuel[X_fuel['Fuel Type'].isna()].count()
X_transmission[X_transmission['Transmission'].isna()].count()


pd.plotting.scatter_matrix(base, figsize=[16,16], alpha=1, marker='o')

y.describe()
X_fuel.describe()
X_transmission.describe()
X_consumption.describe()

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
sns.boxplot(y=base['CO2 Emissions(g/km)'])
plt.title('Identificação de Outliers',)
plt.subplot(1, 2, 2)
sns.histplot(x=base['CO2 Emissions(g/km)'])
plt.title('Distribuição da Emissão de CO2 (g/km)', fontsize=15 )
plt.show()

# instanciar encoder
le = LabelEncoder()
he = OneHotEncoder()

# realizar a transformação
X_fuel_le = le.fit_transform(X_fuel)
print(le.classes_)
X_transmission_le = le.fit_transform(X_transmission)
print(le.classes_)

X_fuel_he = he.fit_transform(X_fuel).toarray()
X_transmission_he = he.fit_transform(X_transmission).toarray()


df_X_transmission = pd.DataFrame(X_transmission_he)
df_X_transmission.rename(columns={0:'A10', 1:'A4', 2:'A5', 3:'A6', 4:'A7', 5:'A8',
                                  6:'A9', 7:'AM5', 8:'AM6', 9:'AM7', 10:'AM8',
                                  11:'AM9', 12:'AS10', 13:'AS4', 14:'AS5', 15:'AS6',
                                  16:'AS7', 17:'AS8', 18:'AS9', 19:'AV', 20:'AV10',
                                  21:'AV6', 22:'AV7', 23:'AV8', 24:'M5', 25:'M6', 26:'M7'}, inplace=True)

df_X_fuel = pd.DataFrame(X_fuel_he)
df_X_fuel.rename(columns={0: 'Fuel Type_D', 1:'Fuel Type_E', 2:'Fuel Type_N',
                          3: 'Fuel Type_X', 4: 'Fuel Type_Z'}, inplace=True)

df_X = pd.merge(df_X_transmission, df_X_fuel, left_index = True, right_index = True, how = "inner")
X = pd.merge(df_X, X_consumption, left_index = True, right_index = True, how = "inner")

scaler_x_reg = StandardScaler()
previsores = scaler_x_reg.fit_transform(X)

# z = (x - u) / s; where u is the mean of the training samples or zero if with_mean=False,
# and s is the standard deviation of the training samples or one if with_std=False.
X.describe()
x_u = scaler_x_reg.mean_
x_s = scaler_x_reg.scale_

X_train, X_test, y_train, y_test = train_test_split(previsores, y,
                                                    test_size = 0.3,
                                                    random_state = 100)
# =============================================================================
# >> TECNICA DE APRENDIZADO DE MAQUINA POR REGRESSAO LINEAR
# =============================================================================
regressor = LinearRegression()
regressor.fit(X_train, y_train)
score_reg_train = regressor.score(X_train, y_train)

previsoes_reg = regressor.predict(X_test)

result_reg = abs(y_test - previsoes_reg)
result_reg.mean()

mae_reg = mean_absolute_error(y_test, previsoes_reg)
mse_reg = mean_squared_error(y_test, previsoes_reg)

plt.scatter(previsoes_reg, y_test,)
plt.title('Emissão CO2 (g/km), Regressão Linear',)
plt.ylabel('Modelo')
plt.xlabel('Dados Teste')
plt.show()

score_reg_test = regressor.score(X_test, y_test)

regressor.intercept_
regressor.coef_

# =============================================================================
# >> TECNICA DE APRENDIZADO DE MAQUINA POR REDES NEURAIS ARTIFICIAIS
# =============================================================================
scaler_x_rna = StandardScaler()
X_rna = scaler_x_rna.fit_transform(X)
scaler_y_rna = StandardScaler()
y_rna = scaler_y_rna.fit_transform(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X_rna, y_rna,
                                                                  test_size = 0.3,
                                                                  random_state = 100)

# activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
# Activation function for the hidden layer.
# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

rna = MLPRegressor(hidden_layer_sizes = (8,), activation='relu')
rna.fit(X_treinamento, y_treinamento)
score_rna = rna.score(X_treinamento, y_treinamento)
score_rna_test = rna.score(X_teste, y_teste)

previsoes_rna = rna.predict(X_teste)

df_y_test_rna = pd.DataFrame(y_teste)
df_previsoes_rna = pd.DataFrame(previsoes_rna)

y_teste = scaler_y_rna.inverse_transform(df_y_test_rna)
previsoes_rna = scaler_y_rna.inverse_transform(df_previsoes_rna)

mae_rna = mean_absolute_error(y_teste, previsoes_rna)
mse_rna = mean_squared_error(y_teste, previsoes_rna)



plt.scatter(previsoes_rna, y_teste, color='red')
plt.title('Emissão CO2 (g/km), Rede Neural Artificial',)
plt.ylabel('Modelo')
plt.xlabel('Dados Teste')
plt.show()

plt.scatter(previsoes_reg, y_test)
plt.scatter(previsoes_rna, y_teste, color='red', alpha=0.5)
plt.title('Emissão CO2 (g/km)',)
plt.ylabel('Modelo')
plt.xlabel('Dados Teste')
plt.legend(['Linear Reg','RNA'])
plt.show()



print('weights ', rna.coefs_)

print('bias ', rna.intercepts_)
print(rna.n_layers_)
loss_curve = rna.loss_curve_

# About Dataset
# CONTEXT
# Hi, I am currently pursuing PGP in Data Science. Recently we were assigned with a project on regression and hypothesis by our Statistics department. While looking for a dataset relevant to my project, I stumbled upon this one.

# CONTENT
# This dataset captures the details of how CO2 emissions by a vehicle can vary with the different features. The dataset has been taken from Canada Government official open data website. This is a compiled version. This contains data over a period of 7 years.
# There are total 7385 rows and 12 columns. There are few abbreviations that has been used to describe the features. I am listing them out here. The same can be found in the Data Description sheet.

# Model
# 4WD/4X4 = Four-wheel drive
# AWD = All-wheel drive
# FFV = Flexible-fuel vehicle
# SWB = Short wheelbase
# LWB = Long wheelbase
# EWB = Extended wheelbase

# Transmission
# A = Automatic
# AM = Automated manual
# AS = Automatic with select shift
# AV = Continuously variable
# M = Manual
# 3 - 10 = Number of gears

# Fuel type
# X = Regular gasoline
# Z = Premium gasoline
# D = Diesel
# E = Ethanol (E85)
# N = Natural gas

# Fuel Consumption
# City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per gallon (mpg)

# CO2 Emissions
# The tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving

# ACKNOWLEDGEMENTS
# The data has been taken and compiled from the below Canada Government official link
# https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6

# QUESTIONS
# Determine or test the influence of different variables on the emission of CO2.
# What are the most influencing features that affect the CO2 emission the most?
# Will there be any difference in the CO2 emissions when Fuel Consumption for City and Highway are considered separately and when their weighted variable interaction is considered?

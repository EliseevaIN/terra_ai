import subprocess
import os
import gdown
from IPython import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

def показать_примеры(старт, финиш):
  #Загружаем датафрейм с сессионного хранилища
  dataframe = pd.read_csv('traff.csv', header=None)
  dataframe.rename(columns={0: "Дата", 1: "Трафик"}, inplace=True)
  
  for i in range(dataframe.shape[0]):
    dataframe.iloc[i, 1] = float(dataframe.iloc[i, 1].replace(',',''))
  plt.figure(figsize=(12, 6)) # Создаем полотно для визуализации
  #Переводим тип данных со столбика с ценами из текста (string) на числа с плавающей запятой (float)
  plt.plot(dataframe['Трафик'][старт:финиш], label = 'Трафик')
  plt.legend()
  plt.show()
  return dataframe

def создать_выборки_трафик(xLen=60, batch_size=20):
  curTime = time()
  #Загружаем данные в переменную dataframe
  dataframe = pd.read_csv('traff.csv', header=None)
  dataframe.rename(columns={0: "Дата", 1: "Трафик"}, inplace=True)
  
  for i in range(dataframe.shape[0]):
    dataframe.iloc[i, 1] = float(dataframe.iloc[i, 1].replace(',',''))
  
  
  #Разделяем на проверочную/обучаюшую выборки с соотношением 20%/80%
  valLen = 300
  trainLen = dataframe.shape[0] - valLen
  xTrain = np.array([dataframe.iloc[:trainLen,1]])
  xTest = np.array([dataframe.iloc[:valLen,1]])

  #Меняем размерность массивов (1, 876) -> (876, 1)
  #                            (1, 157) -> (157, 1)
  xTrain = np.reshape(xTrain, (-1, 1))
  xTest = np.reshape(xTest, (-1, 1))

  xScaler = MinMaxScaler()
  xScaler.fit(xTrain) # Обучаем на обучающей выборке

  xTrain_scaled = xScaler.transform(xTrain)
  xTest_scaled = xScaler.transform(xTest)

  trainDataGen = TimeseriesGenerator(xTrain_scaled, xTrain_scaled, length=xLen, stride=1, batch_size=batch_size)
  testDataGen = TimeseriesGenerator(xTest_scaled, xTest_scaled, length=xLen, stride=1, batch_size=batch_size)
  print('Выборки созданы успешно')
  return trainDataGen, testDataGen, (xTrain, xTest)

def тест_модели_трафика(нейронка, наборы):

  def getPred(model, x1Val, x2Val, xScaler):
    # Предсказываем ответ сети по проверочной выборке
    # И возвращаем исходны масштаб данных, до нормализации
    predVal = xScaler.inverse_transform(model.predict(x1Val.astype('int')))
    xValUnscaled = xScaler.inverse_transform(x2Val.astype('int'))
    return (predVal, xValUnscaled)
  
  def showPredict(start, step, channel, predVal, xValUnscaled):
    plt.figure(figsize=(14,7))
    plt.plot(predVal[start:start+step, 0], label='Прогноз')
    plt.plot(xValUnscaled[start:start+step, channel], label='Базовый ряд')
    plt.xlabel('День')
    plt.ylabel('Количество пользователей')
    plt.legend()
    plt.show()

  testDataGen = TimeseriesGenerator(наборы[1], наборы[1], length=60, stride=1, batch_size=len(наборы[1]))
  x1Val = []
  x2Val = []
  for i in testDataGen:
    x1Val.append(i[0])
    x2Val.append(i[1])
  x1Val = np.array(x1Val)
  x2Val = np.array(x2Val)

  xScaler = MinMaxScaler()
  xScaler.fit(наборы[0])

  print(x1Val.shape)

  for i in range(len(x1Val)):
    (predVal, xValUnscaled) = getPred(нейронка, x1Val[i], x2Val[i], xScaler)
    if i == 0:
      pred = predVal
      xVal = xValUnscaled
    else:
      pred = np.vstack((pred, predVal))
      xVal = np.vstack((xVal, xValUnscaled))

  showPredict(0, len(pred), 0, pred, xVal)
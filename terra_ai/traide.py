import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle 
import os
from tqdm.notebook import tqdm_notebook as tqdm_
from IPython import display
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler # проверить все

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, 
                                     GRU, LSTM, Bidirectional, Conv1D, SeparableConv1D, MaxPooling1D,
                                     Reshape, RepeatVector, SpatialDropout1D)
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import utils
from keras.utils import plot_model
from IPython.display import clear_output


def получить_данные(акции, количество_анализируемых_дней, период_предсказания):
  if акции=='Газпром':
    f = '/content/shares/GAZP_1d_from_MOEX.txt'
  elif акции =='Яндекс':
    f = '/content/shares/YNDX_1d_from_MOEX.txt'
  elif акции =='Полиметаллы':
    f = '/content/shares/POLY_1d_from_MOEX.txt'
  else:
    print('Указанных данных нет')
    return
  data = pd.read_csv(f, sep=",")
  data = clear_data(data, 10)
  x1, y1, x2, y2, инструменты = get_XY(data,количество_анализируемых_дней,период_предсказания)
  return (x1,y1), (x2,y2), (data, инструменты)

def clear_data(data, количество_анализируемых_дней):
    del data['<TICKER>'], data['<PER>'], data['<DATE>'], data['<TIME>'], data['<VOL>']
    data.rename(columns = {'<OPEN>': 'Open'}, inplace=True)
    data.rename(columns = {'<HIGH>': 'High'}, inplace=True)
    data.rename(columns = {'<LOW>': 'Low'}, inplace=True)
    data.rename(columns = {'<CLOSE>': 'Close'}, inplace=True)
    for i in range(1, количество_анализируемых_дней + 1):
      indicator_name = 'Close_chng_%d' % (i)
      data[indicator_name] = data['Close'].pct_change(i) # относительная доходность единицах
    data = data.dropna() # удаляем строки с NaN
    return data
    
def show_data(data):
    data = data[0].copy()
    data1 = data[-3000:]
    fig = plt.figure(figsize=(18,9))    
    ax = fig.add_subplot(111) 
    ax.plot(data1['Close'])
    ax.add_patch( Rectangle((data.shape[0]-2970, data['Close'].max() * 1.1), 
                              2400, (data['Close'].min() - data['Close'].max()) *1.2, 
                              fc ='none',  
                              ec ='g', 
                              lw = 2) ) 
    ax.add_patch( Rectangle((data.shape[0]-470, data['Close'].max() *1.1), 
                              500, (data['Close'].min() - data['Close'].max())*1.2, 
                              fc ='none',  
                              ec ='y', 
                              lw = 2) ) 
    plt.show() 
    
def split_sequence(sequence, Y, количество_анализируемых_дней, период_предсказания):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + количество_анализируемых_дней # находим конечный индекс строки
        if end_ix + (период_предсказания-1) > len(sequence)-1: # cпроверем чтобы не выйти за пределы массива
            break             
        seq_x, seq_y = sequence[i:end_ix], Y[end_ix + (период_предсказания - 1)]
        X.append(seq_x)
        y.append(seq_y) # тк предсказываем только Close
    return array(X), array(y)
    
def get_XY(data,количество_анализируемых_дней,период_предсказания):
    data = data[-3000:].copy()
    test_lvl = .2
    n_steps = количество_анализируемых_дней
    plato_lvl     = 0.008
    indicator_name = 'Close_chng_%d' % (10)# Маркируем направление движения
    a = data[data[indicator_name] < -plato_lvl]
    a.loc[:, indicator_name] = -1.
    a.rename(columns = {indicator_name: 'Down'}, inplace=True)
    b = data[data[indicator_name] >= -plato_lvl]
    b.loc[:, indicator_name] = 0.
    b.rename(columns = {indicator_name: 'Stay'}, inplace=True)
    c = data[data[indicator_name] > plato_lvl]
    c.loc[:, indicator_name] = 1.
    c.rename(columns = {indicator_name: 'Up'}, inplace=True)
    data_UpDown = pd.concat([a['Down'], b['Stay'], c['Up']], axis=1)
    data_UpDown = data_UpDown.fillna(0)
    data_UpDown['Y'] = data_UpDown['Down'] + data_UpDown['Stay'] + data_UpDown['Up']
    
    del data_UpDown['Down'], data_UpDown['Stay'], data_UpDown['Up']        
    categorical_labels = to_categorical(data_UpDown, num_classes = 3)
    data = data.values
    n_train = int(len(data)*test_lvl//n_steps*n_steps) #  то, что уходит в train 
    xTrain = data[:-n_train]
    xTest = data[-n_train:]
    yTrain = categorical_labels[:-n_train]
    yTest = categorical_labels[-n_train:]

    
    """ 
    # Масштабируем только X
    """
    xScaler = RobustScaler()
    xScaler.fit(xTrain)
    xTrain = xScaler.transform(xTrain)
    xTest = xScaler.transform(xTest)

    xTrain, yTrain = split_sequence(xTrain, yTrain,количество_анализируемых_дней,период_предсказания)
    xTest, yTest = split_sequence(xTest, yTest,количество_анализируемых_дней,период_предсказания)
    return xTrain, yTrain, xTest, yTest, xScaler

def model_test(model,
 тестовая_выборка, метки_тестовой_выборки, 
 данные,
 период_предсказания,
 количество_анализируемых_дней
 ):
    скеллер = данные[1]
    x_test = тестовая_выборка
    y_test_org = метки_тестовой_выборки
    conv_test = []
    result = []
    price = []
    for i in tqdm_(range(len(x_test)), desc='Тестрирование модели', ncols=1000): # Проходимся по всем договорам   # Выбираю пример
      x = x_test[i]
      x = np.expand_dims(x, axis=0)
      prediction = model.predict(x) # Распознаём наш пример          
      prediction = np.argmax(prediction) # Получаем индекс самого большого элемента (это итоговая цифра)
      result.append(prediction)
      price.append(скеллер.inverse_transform(x_test[i])[-3][0])
      if prediction == np.argmax(y_test_org[i]):
        conv_test.append('True')
      else:
        conv_test.append('False')             
    from collections import Counter        
    accuracyConv = Counter(conv_test)
    print('Результат теста:')
    print('  * Количество отсчетов:', x_test.shape[0])
    print('  * Правильных предсказаний:', accuracyConv['True'])
    print('  * Ошибочных предсказаний: ', accuracyConv['False'])
    print()
    print('  * Точность предсказаний: ', round(100*accuracyConv['True']/(accuracyConv['True']+accuracyConv['False']),2),'%',sep='')

    print()
    print()
    print()
    print('Примеры распознавания')
    test1 = 0
    test2 = 0
    test3 = 0
    test = 0
    plato_lvl = 0.008
    for i in range(40, 550):
      x = x_test[i]
      x = np.expand_dims(x, axis=0)
      prediction = model.predict(x) # Распознаём наш пример  == {0:stay, 1:up, 2:down}
      signal = np.argmax(prediction) # Получаем индекс самого большого элемента (это итоговая цифра)
      close1 = скеллер.inverse_transform(x_test[i])[-3][0] # опрелеяем текущую цену, подаваемую в нейронку
      close2 = скеллер.inverse_transform(x_test[i+период_предсказания])[-3][0] # опрелеяем текущую цену, подаваемую в нейронку
      if signal == 1 and test1<2:
        if close1 < close2 and abs(close1-close2)/close1 > plato_lvl:
          plt.plot(np.arange(количество_анализируемых_дней+1), price[i-количество_анализируемых_дней:i+1], c='b', label = 'анализируемый период')
          plt.plot(np.arange(количество_анализируемых_дней, количество_анализируемых_дней+период_предсказания + 1),price[i: i + период_предсказания +1 ], c='g', label = 'результат')
          plt.legend()
          plt.show()
          test1+=1
          test+=1
          print('Тест №',test,sep='')
          print('Модель предсказала растущий тренд:')
          print('Текущая цена: ', close1)
          print('Цена через',период_предсказания,'дней(я):',close2)
          print()
          print()
      if signal == 2 and test2<2 and abs(close1-close2)/close1 > plato_lvl:
        if close1 > close2:
          plt.plot(np.arange(количество_анализируемых_дней+1),price[i-количество_анализируемых_дней:i+1], c='b', label = 'анализируемый период')
          plt.plot(np.arange(количество_анализируемых_дней, количество_анализируемых_дней+период_предсказания + 1),price[i: i + период_предсказания +1], c='r', label = 'результат')
          plt.legend()
          plt.show()
          test2+=1
          test+=1
          print('Тест №',test,sep='')
          print('Модель предсказала падающий тренд:')
          print('Текущая цена: ', close1)
          print('Цена через',период_предсказания,'дней(я):',close2)
          print()
          print()
      if signal == 0 and test3<2:
        if abs(close1-close2)/close1 < plato_lvl:
          plt.plot(np.arange(количество_анализируемых_дней+1),price[i-количество_анализируемых_дней:i+1], c='b', label = 'анализируемый период')
          plt.plot(np.arange(количество_анализируемых_дней, количество_анализируемых_дней+период_предсказания + 1),price[i: i + период_предсказания +1], c='y', label = 'результат')
          plt.legend()
          plt.show()
          test3+=1
          test+=1
          print('Тест №',test,sep='')
          print('Модель предсказала нейтральный тренд:')
          print('Текущая цена: ', close1)
          print('Цена через',период_предсказания,'дней(я):',close2)
          print()
          print()
      if test==4:
        break
        
def trading(model, тестовая_выборка, данные):
  x_test = тестовая_выборка
  xScaler = данные [1]
  returns = pd.DataFrame()
  statement = 0 #  {0:in_cash, 1:long, 2:short}
  stock = 0.   # Число акций
  cash = 100000.   # Стартовая сумма капитала
  # -----------------------------------------

  for i in range(len(x_test)):   # Выбираю пример
    x = x_test[i]
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x) # Распознаём наш пример  == {0:stay, 1:up, 2:down}
    signal = np.argmax(prediction) # Получаем индекс самого большого элемента (это итоговая цифра)
    close = xScaler.inverse_transform(x_test[i])[-3][0] # опрелеяем текущую цену, подаваемую в нейронку

    if statement == 0  and  signal == 1:##
      statement = 1
      capital = cash//close * close + cash - cash//close * close
      inv_capital = cash//close * close
      line = pd.DataFrame({'statement':[0], 'signal':[signal], 'close':[close],'stock':[cash//close],  'deal_prise':[close],
                            'long':[close*0.9], 'short':[0],
                          'inv_capital':[inv_capital], 'cash':[cash - cash//close * close],
                          'capital':[capital], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue

    elif statement == 0  and  signal == 2: ##
      statement = 2
      stock = -(cash//close)
      inv_capital = -close*stock
      line = pd.DataFrame({'statement':[0], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[close],
                            'long':[0], 'short':[close*1.1],
                          'inv_capital':[inv_capital],  'cash':[cash - inv_capital], 
                          'capital':[cash], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
      
    elif statement == 0  and  signal == 0:##
      line = pd.DataFrame({'statement':[statement], 'signal':[signal], 'close':[close],'stock':[0],  'deal_prise':[0],
                            'long':[0], 'short':[0],
                          'inv_capital':[0], 'cash':[cash],
                          'capital':[cash], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
      
    elif statement == 2  and  signal == 1: ##
      statement = 1
      ret = (close-returns.iloc[i-1][4])*stock
      capital = -stock*returns.iloc[i-1][4] + returns.iloc[i-1][6] + ret
      stock = capital//close
      inv_capital = close*stock
      cash = capital - inv_capital
      line = pd.DataFrame({'statement':[2], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[close],
                            'long':[close*0.9], 'short':[0],
                          'inv_capital':[inv_capital], 'cash':[cash], 'capital':[capital], 'ret(i)':[ret] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue

    elif statement == 2  and  signal == 2: ##
      stock = returns.iloc[i-1][3]
      cash = returns.iloc[i-1][6]
      inv_capital = -stock*returns.iloc[i-1][4] + (close - returns.iloc[i-1][4])*stock
      capital = inv_capital + cash
      line = pd.DataFrame({'statement':[statement], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[returns.iloc[i-1][4]],
                            'long':[0], 'short':[0],
                          'inv_capital':[inv_capital], 'cash':[cash], 'capital':[capital], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
      
    elif statement == 2  and  signal == 0: ##
      stock = returns.iloc[i-1][3]
      cash = returns.iloc[i-1][6]
      inv_capital = -stock*returns.iloc[i-1][4] + (close - returns.iloc[i-1][4])*stock
      capital = inv_capital + cash
      line = pd.DataFrame({'statement':[statement], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[returns.iloc[i-1][4]],
                            'long':[0], 'short':[0],
                            'inv_capital':[inv_capital], 'cash':[cash], 'capital':[capital], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
      
    elif statement == 1  and  signal == 1:##
      stock = returns.iloc[i-1][3]
      cash = returns.iloc[i-1][6]
      inv_capital = close*stock
      capital = close*stock+cash
      line = pd.DataFrame({'statement':[statement], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[returns.iloc[i-1][4]],
                            'long':[0], 'short':[0],
                            'inv_capital':[inv_capital], 'cash':[cash],  'capital':[capital], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue

    elif statement == 1  and  signal == 2:##
      statement = 2
      stock = returns.iloc[i-1][3]
      ret = (close - returns.iloc[i-1][4]) * stock
      capital = close * stock + returns.iloc[i-1][6]
      stock = -capital//close
      inv_capital = -close*stock
      cash = capital - inv_capital
      line = pd.DataFrame({'statement':[1], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[close],
                            'long':[0], 'short':[close*1.1],
                            'inv_capital':[inv_capital], 'cash':[cash], 'capital':[capital], 'ret(i)':[ret] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
      
    elif statement == 1  and  signal == 0:##
      stock = returns.iloc[i-1][3]
      cash = returns.iloc[i-1][6]
      inv_capital = close*stock
      capital = close*stock+cash
      line = pd.DataFrame({'statement':[statement], 'signal':[signal], 'close':[close], 'stock':[stock], 'deal_prise':[returns.iloc[i-1][4]],
                            'long':[0], 'short':[0],
                            'inv_capital':[inv_capital], 'cash':[cash], 'capital':[capital], 'ret(i)':[0] })
      #clear_output()
      returns = returns.append(line, ignore_index=True)
      #print(returns.shape)
      #print(returns[-5:])
      continue
  return returns
  
def show_long_short_anim(returns, idx_long, idx_short):
  for i in range(len(returns['close'])):    
  #for i in range(100):    
    fig, axs = plt.subplots(2,1,figsize=(24,12))
    axs[0].plot(returns['close'][:i], alpha=0.6)
    l = np.array(idx_long)[np.array(idx_long) < i]
    s = np.array(idx_short)[np.array(idx_short) < i]
    axs[0].plot(l, returns['close'].values[[l]], '^', c='g')
    axs[0].plot(s, returns['close'].values[[s]], 'v', c='r')    
    axs[1].plot(returns['capital'].values[:i])
    plt.show()    
    if i < len(returns['close'])-1:
      clear_output(wait=True)

def show_long_short(returns, idx_long, idx_short):
    plt.figure(figsize=(24,12))
    #plt.xlim(0, 600)
    #plt.ylim(1580, 2100)    
    plt.plot(returns['close'], alpha=0.6)
    l = np.array(idx_long)[np.array(idx_long) < 600]
    s = np.array(idx_short)[np.array(idx_short) < 600]
    plt.plot(l, returns['close'].values[[l]], '^', c='g')
    plt.plot(s, returns['close'].values[[s]], 'v', c='r')    
    plt.show()    
    


def show_capital( returns):
  plt.figure(figsize=(24,8))
  plt.plot(returns['capital'])
  plt.show()

def traiding(нейронка, тестовая_выборка, данные, тип):
    returns = trading(нейронка, тестовая_выборка, данные).copy()
    short = returns['short'].values
    short = short.astype(bool)
    long = returns['long'].values
    long = long.astype(bool)
    idx_long = np.where(long) 
    idx_short = np.where(short)
    if тип=='результат':
        show_long_short(returns, idx_long, idx_short)
        show_capital(returns)
    elif тип=='процесс':
        show_long_short_anim(returns, idx_long, idx_short)
        
        
        

        
from . import обнаружение
from IPython import display
import gym # здесь лежат все модели для 
from gym import logger as gymlogger # загрузим библиотеку для изменения параметра вывода ошибок
from gym.wrappers import Monitor # библиотека для обертки видео из хранилища в ячейку
gymlogger.set_level(40) # для правильного отображения ошибок
import numpy as np # библиотека массивов numpy
import random # библиотека для генерации случайных значений
import matplotlib # библиотека для визуализации процессов
import matplotlib.pyplot as plt # библиотека для построения графика
import glob # расширение для использования Unix обозначений при задании пути к файлу
import io # библиотека для работы с потоковыми данными
import base64 # расширение для преобразования в формат base64 (универсальный формат хранения сырых изображений в виде набора электрических сигналов)
from IPython.display import HTML # библиотека для кодирования в код HTML
import time # библиотека для расчета времени обучения
from IPython import display as ipythondisplay # для работы с "сырым" форматом (набор сигналов, а не пиксели)
from pyvirtualdisplay import Display # для создания окна дисплея

def загрузить_модули():  
  print('Выполняется загрузка и установка модулей')
  print('Это может занять несколько минут...')
  обнаружение.выполнить_команду('apt install swig cmake libopenmpi-dev zlib1g-dev > /dev/null 2>&1')
  обнаружение.выполнить_команду('pip -q install stable-baselines==2.5.1 box2d box2d-kengz > /dev/nul 2>&1')
  обнаружение.выполнить_команду('pip -q install gym pyvirtualdisplay > /dev/null 2>&1')
  обнаружение.выполнить_команду('pip -q install xvfbwrapper > /dev/null 2>&1')
  обнаружение.выполнить_команду('apt-get update > /dev/null 2>&1')
  обнаружение.выполнить_команду('sudo apt-get install xvfb > /dev/null 2>&1')
  обнаружение.выполнить_команду('apt-get install xdpyinfo > /dev/null 2>&1')
  display.clear_output(wait=True)
  print('Все модули установлены и готовы к работе')

def создать_дисплей(ширина, высота):
  display = Display(visible=0, size=(ширина, высота))
  display.start()
  print('Виртуальный дисплей создан')

def показать_пример(задача):
  def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
      mp4 = mp4list[0]
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)
      ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                  loop controls style="height: 400px;">
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii'))))
    else: 
      print("Could not find video")
    

  def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

  env = gym.make('LunarLander-v2')
  
  numBots = 100
  popul = [] # здесь будет лежать популяция

  for i in range(numBots):  
    bot = np.random.random((8,4))
    popul.append(bot)

  env = wrap_env(env) # оборачиваем наше окружения для записи видео
  env.seed(0) # симуляция одного и того же окружения
  observation = env.reset() # обнуляем вектор наблюдения
  i=0
  action = 0
  while True:

      env.render() # Рендер окружения
      # В качестве действия будем подавать значения нашего лучшего бота
      observation,reward,done,_ = env.step(action) # остлеживаем все параметры для подсчета функции значения
      result = np.dot(observation,popul[np.random.randint(0, numBots-1)])
      action = np.argmax(result)
      if done: 
        break;

  
  env.close()
  show_video()

def обучить_алгоритм(размер_популяции, количество_эпох, коэфицент_мутации, количество_выживших):
  
  '''
    Функция получения выжившей популяции
        Входные параметры:
        - popul - наша популяция
        - val - текущие значения
        - nsurv - количество выживших
        - reverse - указываем требуемую операцию поиска результата: максимизация или минимизация
  '''
  def getSurvPopul(
          popul,
          val,
          nsurv,
          reverse
          ):
      newpopul = [] # Двумерный массив для новой популяции

      sval= sorted(val, reverse=reverse) # Сортируем зачения в val в зависимости от параметра reverse
      for i in range(nsurv): # Проходимся по циклу nsurv-раз (в итоге в newpopul запишется nsurv-лучших показателей)
          index = val.index(sval[i]) # Получаем индекс i-того элемента sval в исходном массиве val
          newpopul.append(popul[index]) # В новую папуляцию добавляем элемент из текущей популяции с найденным индексом
      return newpopul, sval # Возвращаем новую популяцию (из nsurv элементов) и сортированный список

  '''
    Функция получения родителей
        Входные параметры:
        - curr_popul - текущая популяция
        - nsurv - количество выживших
'''
  def getParents(
          curr_popul,
          nsurv
          ):   
      indexp1 = random.randint(0, nsurv - 1) # Случайный индекс первого родителя в диапазоне от 0 до nsurv - 1
      indexp2 = random.randint(0, nsurv - 1) # Случайный индекс второго родителя в диапазоне от 0 до nsurv - 1    
      botp1 = curr_popul[indexp1] # Получаем первого бота-родителя по indexp1
      botp2 = curr_popul[indexp2] # Получаем второго бота-родителя по indexp2 
    
      return botp1,botp2 # Возвращаем обоих полученных ботов

  '''
    Функция смешивания (кроссинговера) двух родителей
        Входные параметры:
        - botp1 - первый бот-родитель
        - botp2 - второй бот-родитель
        - j - номер компонента бота
'''

  def crossPointFrom2Parents(
          botp1,
          botp2, 
          j
          ):
      pindex = np.random.random() # Получаем случайное число в диапазоне от 0 до 1
      
      #Если pindex меньше 0.5, то берем значения от первого бота, иначе от второго
      if pindex < 0.5:
          x = botp1[j]
      else:
          x = botp2[j]
      return x # Возвращаем значние бота  

  '''
    Функция расчета вознаграждения за эпизод
        Входные параметры:
        - popul - популяция ботов
'''


  def countValue(
      popul
      ):
    
    action=0  # генерируем первое  действие случайным из пространства действий
                                      # 0 - ничего не делать 
                                      # 1 - запустить двигатель с левой ориентацией
                                      # 2 - запустить двигатель по центру
                                      # 3 - запустить двигатель с правой ориентацией

    reward_list=[] # здесь будет сумма вознаграждений для каждого эпизода

    for bot in popul: # проходимся по каждому боту в популяции
      
      env.reset() # И обновлять окружение
      i=0 
      tmp=0
      done=False
      while done!=True:
        observation,reward,done,_ = env.step(action) # остлеживаем все параметры для подсчета функции значения
        result = np.dot(observation,bot) # матрично перемножаем бота и вектор наблюдения для предсказания следующего движения
        action = np.argmax(result) # максимальный аргумент - наше движение
        tmp+=reward
      reward_list.append(tmp) # Функция, по которой будет вычисляться "успех текущей симуляции"
      
    return reward_list


  env = gym.make('LunarLander-v2')
  total=[] # Для построения графика
  n = размер_популяции
  nsurv = количество_выживших
  nnew = n-nsurv # количество новых
  epohs = количество_эпох
  mut = коэфицент_мутации
  curr_time = time.time()
  
  global popul
  global newpopul

  numBots = 100
  popul = [] # здесь будет лежать популяция

  for i in range(numBots):  
    bot = np.random.random((8,4))
    popul.append(bot)

  for it in range(epohs): # создали список списков всех значений по эпохам
    val = countValue(popul) # считаем успех каждого из ботов
    newpopul, sval = getSurvPopul(popul, val, nsurv,1) # получили популяцию выживших, нас интересует бот с максимальным успехом, поэтому reverse = 1
    print('Выполняется эпоха №' '{:>2} {:>3}'.format(it,'Время на выполнение:' ), '{:>4}'.format(np.round(time.time() - curr_time, 1)), 'сек,', '{:>30}'.format('Награда для 3-x лучших ботов'), np.array2string(np.round(sval[0:3], 2).astype('int'),  precision=2,suppress_small=True)) # Выводим время на операцию, среднее значение и 3 лучших ботов
    total.append(sval[0]) # заносим самого лучшего бота в список для построения графика эволюции 
    curr_time = time.time() # Обновляем текущее время

  # проходимся по новой популяции
    for k in range(nnew):
      
      # вытаскиваем новых родителей
      botp1, botp2 = getParents(newpopul, nsurv) 
      newbot = [] # здесь будет новый бот
      
      for j in range(len(botp1)): # боты-родители одинаковой длины, будем проходиться по каждому элементу родителя
        x = crossPointFrom2Parents(botp1, botp2,j) # скрещиваем
        for t in range(4):
          if random.random()<mut:
            x[t] += random.random()*1e-1
        newbot.append(x) # закидываем элемент в бота
      newpopul.append(newbot) # добавляем бота
      
    popul = newpopul # вывести список на эпоху
    popul = np.array(popul) # для того, чтобы можно было легко вытащить индексы условием, преобразуем в numpy массив
  plt.plot(total)
  print('Среднее количество баллов', round(np.mean(total), 2))  

def посадить_корабль():
  def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
      mp4 = mp4list[0]
      video = io.open(mp4, 'r+b').read()
      encoded = base64.b64encode(video)
      ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                  loop controls style="height: 400px;">
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>'''.format(encoded.decode('ascii'))))
    else: 
      print("Could not find video")
      
  env = gym.make('LunarLander-v2')
  def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env
  
  env = wrap_env(env) # оборачиваем наше окружения для записи видео
  observation = env.reset() # обнуляем вектор наблюдения
  i=0
  action = 0
  reward = 0
  while reward!=100:
    
    env.render() # Рендер окружения
    # В качестве действия будем подавать значения нашего лучшего бота
    observation,reward,done,_ = env.step(action) # остлеживаем все параметры для подсчета функции значения
    result = np.dot(observation, popul[0])
    action = np.argmax(result)
    if done:       
      observation = env.reset() # обнуляем вектор наблюдения

  env.close()
  show_video()
  
  if reward == 100:
    print('Поздравляем, вы посадили корабль! ', 'Награда: ', reward, 'баллов')
  elif reward == -100:
    print('Корабль разбился, попробуйте еще раз! ', 'Награда: ', reward, 'баллов')

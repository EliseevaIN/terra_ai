import time, os, random
from tensorflow.keras.preprocessing import image # Импортируем модуль image для работы с изображениями
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
import numpy as np
from tqdm import tqdm
from IPython import display
from tqdm.notebook import tqdm_notebook as tqdm_
from sklearn.model_selection import train_test_split
from PIL import Image # подключаем модуль Image для работы с изображениями
import matplotlib.pyplot as plt # Импортируем модуль pyplot библиотеки matplotlib для построения графиков

# Функция преобразования пикселя сегментированного изображения в индекс
def color2index(color):
    class_=0 # По умолчанию задаем 0-ой класс (фон)

    # Если цвет пикселя красный (0-ой канал изображения)
    if color[0] > 20: 
      class_ = 1 # Меняем номер класса на 1-ый
    return class_ # Возвращаем номер класса

def index2color(index2):
    index = np.argmax(index2)
    color=[]
    if index == 0:
        color = [0, 0, 0]  # фон
    elif index == 1:
        color = [255, 0, 0]  # самолет
    return color 
    
# Функция преобразования индекса в цвет пикселя
def index2color2(index2):
  index = np.argmax(index2) # Получаем индекс максимального элемента
  color=[]
  if index == 0: color = [100, 100, 100]  # пол
  elif index == 1: color = [0, 0, 100]  # потолок
  elif index == 2: color = [0, 100, 0]  # стена
  elif index == 3: color = [100, 0, 0]  # проем, дверь, окно
  elif index == 4: color = [0, 100, 100]  # колонна, лестница, внешний мир, перила, батарея, инвентарь, источники света, провода, балка
  elif index == 5: color = [100, 0, 100]  # люди
  elif index == 6: color = [0, 0, 0]  # остальное
  return color # Возвращаем цвет пикслея


# Функция перевода индекса пиксея в to_categorical
def rgb2ohe(y): 
  y2 = y.copy() # Создаем копию входного массива
  y = y.reshape(y.shape[0] * y.shape[1], 3) # Решейпим в двумерный массив
  yt = [] # Создаем пустой лист

  # Проходим по всем трем канала изображения
  for i in range(len(y)): 
    # Переводим пиксели в индексы и преобразуем в OHE
    yt.append(utils.to_categorical(color2index(y[i]), num_classes=2))

  yt = np.array(yt) # Преобразуем в numpy
  yt = yt.reshape(y2.shape[0], y2.shape[1], 2) # Решейпим к исходному размеру
  
  return yt # Возвращаем сформированный массив

def get_images(**kwargs):
  if kwargs['оригиналы'] == 'Самолеты':
    images_airplane = [] # Создаем пустой список для оригинальных изображений
    cur_time = time.time() # Засекаем текущее время
    # Проходим по всем файлам в каталоге по указанному пути     
    for filename in sorted(os.listdir(kwargs['оригиналы'])): 
      # Читаем очередную картинку и добавляем ее в список с указанным target_size                                                      
      images_airplane.append(image.load_img(os.path.join(kwargs['оригиналы'],filename), 
                                              target_size=kwargs['размер']))     
    # Отображаем время загрузки картинок обучающей выборки
    segments_airplane = []
    # Проходим по всем файлам в каталоге по указанному пути     
    for filename in sorted(os.listdir(kwargs['сегменты'])): 
      # Читаем очередную картинку и добавляем ее в список с указанным target_size                                                      
      segments_airplane.append(image.load_img(os.path.join(kwargs['сегменты'],filename),
                                              target_size=kwargs['размер'])) 
  elif kwargs['оригиналы'] == 'diseases/origin':
    print('Обработка изображений кожных заболеваний')
    print('Это может занять несколько минут...')
    images_airplane = [] # Создаем пустой список для оригинальных изображений
    cur_time = time.time() # Засекаем текущее время
    # Проходим по всем файлам в каталоге по указанному пути     
    for d in kwargs['классы']:    
      for filename in sorted(os.listdir(kwargs['оригиналы'] + '/' + d)): 
        # Читаем очередную картинку и добавляем ее в список с указанным target_size                                                      
        images_airplane.append(image.load_img(os.path.join(kwargs['оригиналы'] +'/' + d,filename), 
                                                target_size=kwargs['размер']))     
    # Отображаем время загрузки картинок обучающей выборки
    
    segments_airplane = []
    for d in kwargs['классы']:
      # Проходим по всем файлам в каталоге по указанному пути     
      for filename in sorted(os.listdir(kwargs['сегменты'] + '/' + d)):
        # Читаем очередную картинку и добавляем ее в список с указанным target_size                                                      
        segments_airplane.append(image.load_img(os.path.join(kwargs['сегменты'] + '/' +d,filename),
                                                target_size=kwargs['размер'])) 
    display.clear_output(wait=True)
    print('Обработка изображений кожных заболеваний (Готово)')
  return images_airplane, segments_airplane
    
  # Отображаем время загрузки картинок обучающей выборки  
 
# Функция формирования yTrain
def yt_prep(data, num_classes):
  yTrain = [] # Создаем пустой список под карты сегментации
  for i in tqdm_(range(len(data)), desc='Обработка изображений', ncols=1000):  
    seg = data[i]
    y = image.img_to_array(seg) # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
    y = rgb2ohe(y) # Получаем OHE-представление сформированного массива    
    yTrain.append(y) # Добавляем очередной элемент в yTrain
  return np.array(yTrain) # Возвращаем сформированный yTrain

def create_xy(images_airplane, segments_airplane):
  cur_time = time.time() # Засекаем текущее время
  yTrain = yt_prep(segments_airplane, 2) # Формируем yTrain
  xTrain = [] # Создаем пустой список под xTrain
  for img in images_airplane: 
      x = image.img_to_array(img) # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
      xTrain.append(x) # Добавляем очередной элемент в xTrain
  xTrain = np.array(xTrain) # Переводим в numpy
  # Разделяем данные на обучающую и проверочную выборки
  x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size = 0.1)
  print('Данные загружены. Обработано:', xTrain.shape[0],'изображений')
  print('Время обработки: ', round(time.time() - cur_time, 2),'c')
  return (x_train, y_train), (x_test, y_test)

def add_mask(img, seg):
  segment = seg.copy().convert('RGBA')
  mask = np.array(segment)
  mask[mask[:,:,0] <= 10] = [0, 0, 0, 0]
  mask[mask[:,:,0] > 10] = [0, 150, 0, 200]
  img2 = Image.fromarray(img.astype('uint8'))
  img_new = Image.fromarray(mask).convert('RGBA')
  img2.paste(img_new, (0, 0),img_new)
  
  return img2


def show_sample(**kwargs):
  if 'количество' in kwargs:
    n = kwargs['количество'] # Количество выводимых случайных картинок
  else:
     n = 5
  fig, axs = plt.subplots(2, n, figsize=(25, 7)) #Создаем полотно из n графиков
  # Выводим в цикле n случайных изображений
  for i in range(n):
    idx = np.random.randint(len(kwargs['оригиналы']))
    img = kwargs['оригиналы'][idx] # Выбираем случайное фото для отображения
    axs[0,i].imshow(img) # Отображаем фото
    axs[0,i].axis('off')  # Отключаем оси
    img2 = kwargs['сегментированные_изображения'][idx] # Выбираем случайное фото для отображения
    axs[1,i].imshow(img2) # Отображаем фото
    axs[1,i].axis('off')  # Отключаем оси
  plt.show() #Показываем изображения

def тест_модели(мод, тестовые_изображения, **kwargs):
  count = 5
  if 'количество_классов' in kwargs:
    n_classes = kwargs['количество_классов']
    cnt = 2
  else:
    n_classes = 2
    cnt = 3
  x_val = тестовые_изображения
  indexes = np.random.randint(0, len(x_val), count) # Получаем count случайных индексов  
  fig, axs = plt.subplots(cnt, count, figsize=(25, 10)) #Создаем полотно из n графиков
  for i,idx in enumerate(indexes): # Проходим по всем сгенерированным индексам
    predict = np.array(мод.predict(x_val[idx][None,...])) # Предиктим картику
    pr = predict[0] # Берем нулевой элемент из перидкта
    pr1 = [] # Пустой лист под сегментированную картинку из predicta
    pr = pr.reshape(-1, n_classes) # Решейпим предикт
    for k in range(len(pr)): # Проходим по всем уровням (количесвто классов)
        if 'количество_классов' in kwargs:
            pr1.append(index2color2(pr[k])) # Переводим индекс в писксель
        else:
            pr1.append(index2color(pr[k])) # Переводим индекс в писксель
    pr1 = np.array(pr1) # Преобразуем в numpy
    pr1 = pr1.reshape(x_val[0].shape) # Решейпим к размеру изображения
    img = Image.fromarray(pr1.astype('uint8')) # Получаем картику из предикта
    axs[0,i].imshow(img.convert('RGBA')) # Отображаем на графике в первой линии
    axs[0,i].axis('off')
    axs[1,i].imshow(x_val[idx]/255) # Отображаем на графике в третьей линии оригинальное изображение        
    axs[1,i].axis('off')
    if 'количество_классов' not in kwargs:
        im = add_mask(x_val[idx], img.convert('RGBA'))
        axs[2,i].imshow(im) # Отображаем на графике в третьей линии оригинальное изображение        
        axs[2,i].axis('off')
  plt.show() 

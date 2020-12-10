from . import сегментация, повышение_размерности, обнаружение, обработка_текста
import subprocess
from subprocess import STDOUT, check_call
import os, time, random, re,pymorphy2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from IPython import display
from tensorflow.keras.utils import to_categorical, plot_model # Полкючаем методы .to_categorical() и .plot_model()
from tensorflow.keras import datasets # Подключаем набор датасетов
import importlib.util, sys, gdown
from tqdm.notebook import tqdm_notebook as tqdm_
from PIL import Image


###
###                 ЗАГРУЗКА ДАННЫХ
###
def загрузить_базу(база = '', справка = False):
  print('Загрузка данных')
  print('Это может занять несколько минут...')  
  if база == 'MNIST':
    (x1, y1), (x2, y2) = загрузить_базу_МНИСТ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу рукописных цифр MNIST. \nБаза состоит из двух наборов данных: обучающего (60тыс изображений) и тестовго (10тыс изображений).')
      print('Размер каждого изображения: 28х28 пикселей')
    return (x1, y1), (x2, y2)
  
  elif база == 'АВТО':
    загрузить_базу_АВТОМОБИЛИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу с изображениями марок автомобилей. \nБаза состоит из двух марок: Феррари и Мерседес')
      print('Количество изображений в базе: 2249') 

  elif база == 'АВТО-3':
    загрузить_базу_АВТОМОБИЛИ_3()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу с изображениями марок автомобилей. \nБаза состоит из трех марок: Феррари, Мерседес и Рено')
      print('Количество изображений в базе: 3429')

  elif база == 'САМОЛЕТЫ':
    x,y = загрузить_базу_САМОЛЕТЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу самолетов. \nБаза состоит из оригинальных изображений и соовтетствующих им размеченных сегментированных изображений.')
      print('Количество изображений в базе: 981')    
    return x, y

  elif база == 'КОЖНЫЕ ЗАБОЛЕВАНИЯ':
    загрузить_базу_БОЛЕЗНИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу кожных заболеваний. \nБаза состоит из оригинальных изображений и соовтетствующих им размеченных сегментированных изображений.')
      print('Количество категорий заболеваний: 10 (Акне, Витилиго, Герпес, Дерматит, Лишай, Невус, Псориаз, Сыпь, Хлоазма, Экзема)')    
      print('Количество изображений в базе: 981')    
  
  elif база == 'ПОВЫШЕНИЕ РАЗМЕРНОСТИ':
    x, y = загрузить_базу_HR()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Вы скачали базу изображений для задачи повышения размерности')
      print('База содержит изображения высокого качества и соответствующие им изображения низкого качества')
    return x, y

  elif база == 'ОБНАРУЖЕНИЕ ЛЮДЕЙ':
    загрузить_базу_ЛЮДЕЙ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Загружена размеченная база изображений для обнаружения людей')      
    print()
    print('ВНИМАНИЕ!!! Были установлены дополнительные библиотеки. Необходимо перезапустить среду для продолжения работы')
    print('Выберите пункт меню Runtime/Restart runtime и нажмите «Yes»')
    print('После этого сделайте повторный запуск ячейки: import terra_ai')
  
  elif база == 'СИМПТОМЫ ЗАБОЛЕВАНИЙ':
    загрузить_базу_ЗАБОЛЕВАНИЯ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Загружена база симптомов заболеваний')      
    print()
  elif база == 'ДИАЛОГИ':
    x,y = загрузить_базу_ДИАЛОГИ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Загружена база диалогов')
      print('Количество пар вопрос-ответ: 50 тысяч')
    print()
    return x,y
  elif база == 'ДОГОВОРА':
    x = загрузить_базу_ДОГОВОРА()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Загружена база договоров')
      print('База размечена по 6 категориям: Условия Запреты - Стоимость - Деньги - Сроки - Неустойка')
# s6 Всё про адреса и геолокации
    print()
    return x
  elif база == 'КВАРТИРЫ':
    загрузить_базу_КВАРТИРЫ()
    display.clear_output(wait=True)
    print('Загрузка данных завершена \n')
    if справка:
      print('Загружена база квартир')
    print()
  else:
    display.clear_output(wait=True)
    print('Указанная база не найдена \n')

def загрузить_базу_МНИСТ():
  (x_train_org, y_train_org), (x_test_org, y_test_org) = datasets.mnist.load_data() # Загружаем данные набора MNIST
  return (x_train_org, y_train_org), (x_test_org, y_test_org)

def загрузить_базу_АВТОМОБИЛИ():
  url = 'https://storage.googleapis.com/aiu_bucket/car_2.zip'
  output = 'car.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "car.zip",
      куда = "/content/автомобили"
  )

def загрузить_базу_КВАРТИРЫ():
  url = 'https://storage.googleapis.com/aiu_bucket/moscow.csv'
  output = 'moscow.csv' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

def загрузить_базу_ДИАЛОГИ():
  #обнаружение.выполнить_команду('mkdir content')
  url = 'https://storage.googleapis.com/aiu_bucket/dialog.txt'
  output = 'dialog.txt' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  f = open('dialog.txt', 'r', encoding='utf-8')
  text= f.read()
  text = text.replace('"','')
  text = text.split('\n')
  question = text[::3]
  answers = text[1::3]
  return question[:-1], answers
  
def загрузить_базу_АВТОМОБИЛИ_3():
  url = 'https://storage.googleapis.com/aiu_bucket/car.zip'
  output = 'car.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "car.zip",
      куда = "/content/автомобили"
  )

def загрузить_базу_ЗАБОЛЕВАНИЯ():
  url = 'https://storage.googleapis.com/aiu_bucket/symptoms.zip'
  output = 'symptoms.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "symptoms.zip",
      куда = "content/"
  )
  
def загрузить_базу_ЛЮДЕЙ():
  url = 'https://github.com/ultralytics/yolov5/archive/master.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )
  # Скачиваем и распаковываем архив
  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )

  url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
  output = 'tmp.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  распаковать_архив(
      откуда = "tmp.zip",
      куда = "/content"
  )
  обнаружение.выполнить_команду('echo y|pip uninstall albumentations > /dev/null')
  обнаружение.выполнить_команду('pip install -q --no-input -U git+https://github.com/albumentations-team/albumentations > /dev/null')
  обнаружение.выполнить_команду('pip install -q -U PyYAML')
  
  
def загрузить_базу_ДОГОВОРА():
  url = 'https://storage.googleapis.com/aiu_bucket/docs.zip'
  output = 'Договоры.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL

  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "Договоры.zip",
      куда = "/content/Договоры"
      )
  def readText(fileName):
    f = open(fileName, 'r', encoding='utf-8') #Открываем наш файл для чтения и считываем из него данные 
    text = f.read() #Записываем прочитанный текст в переменную 
    delSymbols = ['\n', "\t", "\ufeff", ".", "_", "-", ",", "!", "?", "–", "(", ")", "«", "»", "№", ";"]
    for dS in delSymbols: # Каждый символ в списке символов для удаления
      text = text.replace(dS, " ") # Удаляем, заменяя на пробел
    text = re.sub("[.]", " ", text) 
    text = re.sub(":", " ", text)
    text = re.sub("<", " <", text)
    text = re.sub(">", "> ", text)
    text = ' '.join(text.split()) 
    return text # Возвращаем тексты
  def text2Words(text):
    morph = pymorphy2.MorphAnalyzer() # Создаем экземпляр класса MorphAnalyzer
    words = text.split(' ') # Разделяем текст на пробелы
    words = [morph.parse(word)[0].normal_form for word in words] #Переводим каждое слово в нормалную форму  
    return words # Возвращаем слова   
  directory = 'Договоры/Договора432/' # Путь к папке с договорами
  agreements = [] # Список, в который запишем все наши договоры
  for filename in os.listdir(directory): # Проходим по всем файлам в директории договоров
    try:    
        txt = readText(directory + filename) # Читаем текст договора
        if txt != '': # Если текст не пустой
          agreements.append(readText(directory + filename)) # Преобразуем файл в одну строку и добавляем в agreements
    except:
        continue
  words = [] # Здесь будут храниться все договора в виде списка слов
  curTime = time.time() # Засечем текущее время
  for i in tqdm_(range(len(agreements)), desc='Обработка догововров', ncols=1000): # Проходимся по всем договорам
    words.append(text2Words(agreements[i])) # Преобразуем очередной договор в список слов и добавляем в words
  wordsToTest = words[-10:] # Возьмем 10 текстов для финальной проверки обученной нейронной сети 
  words = words[:-10] # Для обученающей и проверочной выборок возьмем все тексты, кроме последних 10
  display.clear_output(wait=True)
  return (agreements, words, wordsToTest)  #, agreements


def загрузить_базу_САМОЛЕТЫ():
  url = 'https://storage.googleapis.com/aiu_bucket/%D0%A1%D0%B0%D0%BC%D0%BE%D0%BB%D0%B5%D1%82%D1%8B.zip' # Указываем URL-файла
  output = 'самолеты.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "самолеты.zip",
      куда = "/content"
  )
  url = 'https://storage.googleapis.com/aiu_bucket/segment.zip' # Указываем URL-файла
  output = 'сегменты.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "сегменты.zip",
      куда = "/content"
  )
  display.clear_output(wait=True)  
  # Обрабатываем скаченные изображения
  x, y = изображения, сегментированные_изображения = обработка_изображений(
      оригиналы = 'Самолеты',
      сегменты = 'Сегменты',
      размер = (176, 320)
  )
  return x, y

def загрузить_базу_БОЛЕЗНИ():
  url = 'https://storage.googleapis.com/aiu_bucket/origin.zip'          
  output = 'diseases.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "diseases.zip",
      куда = "/content/diseases"
  )
  url = 'https://storage.googleapis.com/aiu_bucket/segmentation.zip'
  output = 'segm.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "segm.zip",
      куда = "/content/diseases"
  )

def загрузить_базу_HR():
  # Скачиваем и распаковываем архив в колаб по ссылке
  url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip'          
  output = 'DIV2K_valid_LR_bicubic_X4.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "DIV2K_valid_LR_bicubic_X4.zip",
      куда = "/content/div2k"
  )
  url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
  output = 'DIV2K_valid_HR.zip' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  распаковать_архив(
      откуда = "DIV2K_valid_HR.zip",
      куда = "/content/div2k"
  )
  pathHR = '/content/div2k/DIV2K_valid_HR/'
  pathLR = '/content/div2k/DIV2K_valid_LR_bicubic/X4/'
  imagesList = [os.listdir(pathHR), os.listdir(pathLR)] # Список из корневых каталогов изображений, их два - hr и lr
  train_hr = []
  train_lr = []

  # В цикле проходимся по каталогу с hr изображениями
  for image in sorted(imagesList[0]):
      img = Image.open(pathHR + image)
      train_hr.append(img) # Добавляем все изображения в список

  # В цикле проходимся по каталогу с lr изображениями
  for image in sorted(imagesList[1]):
      img = Image.open(pathLR + image)
      train_lr.append(img) # Добавляем все изображения в список
  return train_hr, train_lr

###
###                    ДЕМОНСТРАЦИЯ ПРИМЕРОВ
###

def показать_примеры(**kwargs):
  if 'вопросы' in kwargs:  
    count = 1
    if 'количество' in kwargs:
        count = kwargs['количество']
    questions = kwargs['вопросы']
    answers = kwargs['ответы']
    for i in range(count):
        print()
        idx = np.random.randint(len(questions)-1)
        print('Вопрос:', questions[idx])
        print('Ответ:', answers[idx])
  if 'путь' in kwargs:    
    kwargs['путь'] = '/content'+kwargs['путь']
    count = len(os.listdir(kwargs['путь']))    
    fig, axs = plt.subplots(1, count, figsize=(25, 5)) #Создаем полотно из 3 графиков
    for i in range(count): #Проходим по всем классам
      car_path = kwargs['путь'] + '/' + sorted(os.listdir(kwargs['путь']))[i] + '/'#Формируем путь к выборке
      img_path = car_path + np.random.choice(os.listdir(car_path)) #Выбираем случайное фото для отображения
      axs[i].imshow(image.load_img(img_path, target_size=(108, 192))) #Отображение фотографии
      axs[i].axis('off') # отключаем оси
    plt.show() #Показываем изображения
  if 'база' in kwargs:    
    if kwargs['база'] == 'симптомы':        
        path = 'content/Болезни/'
        text = []
        classes = []
        n = 0
        codecs_list = ['UTF-8', 'Windows-1251']

        for filename in os.listdir(path): # Проходим по всем файлам в директории договоров
            n +=1
            for codec_s in codecs_list:
                try:
                    text.append(обработка_текста.readText(path+filename, codec_s)) # Преобразуем файл в одну строку и добавляем в agreements
                    classes.append(filename.replace(".txt", ""))
                    break
                except UnicodeDecodeError:
                    print('Не прочитался файл: ', path+currdir+'/'+filename, codec_s)
                else:
                    next 
        nClasses = len(classes) #определяем количество классов
        print('В данной базе содержатся симптомы следующих заболеваний:')
        print(classes)        
        n = np.random.randint(10)
        print()
        print('Пример симптомов случайного заболевания:')
        print('Заболевание: ', classes[n])
        print('Симптомы:')
        print('     *', text[n][:100]) # Пример первых 100 символов первого документа
        
        
        return classes, nClasses
        
  elif ('изображения' in kwargs):
    if 'метки' in kwargs:
      count = kwargs['метки'].max() # Задачем количество примеров
    elif 'количество' in kwargs:
      count = kwargs['количество'] # Задачем количество примеров
    else:
      count = 5
    f, axs = plt.subplots(1,count,figsize=(22,5)) # Создаем полотно для визуализации
    idx = np.random.choice(kwargs['изображения'].shape[0], count) # Получаем 5 случайных значений из диапазона от 0 до 60000 (x_train_org.shape[0])
    for i in range(count):
      axs[i].imshow(kwargs['изображения'][idx[i]], cmap='gray') # Выводим изображение из обучающей выборки в черно-белых тонах
      axs[i].axis('off')
    plt.show()
  elif ('оригиналы' in kwargs):
    показать_примеры_сегментации(**kwargs)
  elif ('изображения_низкого_качества' in kwargs):
    показать_примеры_HR(**kwargs)
  elif ('база' in kwargs):
    if(kwargs['база'] == 'обнаружение'):
      обнаружение.показать_примеры()

def показать_примеры_HR(**kwargs):
  lr = kwargs['изображения_низкого_качества']
  hr = kwargs['изображения_высокого_качества']
  '''
  показать_примеры_изображений - функция вырезает небольшие кусочки из hr и lr изображений и выводит в масштабе на экран
  вход:
    lr, hr - списки с lr, hr изображениями соответственно
  '''
  n = 3 # Указываем кол-во отображаемых пар изображений
  fig, axs = plt.subplots(n, 2, figsize=(10, 15)) # Задаем размера вывода изображения

  for i in range(n): # В цикле попарно выводим изображения
    ind = random.randint(0, len(lr)) # Выбираем случайный индекс изображения

    area = (100, 100, 200, 200) # Задаем координаты точек для вырезания участка из изображения низкого качества
    cropped_lr = lr[ind].crop(area) # Вырезаем участок
    area = (400, 400, 800,800) # Задаем координаты точек для вырезания участка из изображения высокого качества
    cropped_hr = hr[ind].crop(area) # Вырезаем участок

    axs[i,0].axis('off')
    axs[i,0].imshow(cropped_lr) # Отображаем lr изображение
    axs[i,0].set_title('Низкое качество', fontsize=30)
    axs[i,1].axis('off')
    axs[i,1].imshow(cropped_hr) # Отображаем hr изображение
    axs[i,1].set_title('Высокое качество', fontsize=30)
  plt.show() #Показываем изображения

###
###                СОЗДАНИЕ ВЫБОРОК
###

def создать_выборки(путь, размер, коэф_разделения=0.9):  
  путь = '/content'+путь
  x_train = [] # Создаем пустой список, в который будем собирать примеры изображений обучающей выборки
  y_train = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)
  x_test = [] # Создаем пустой список, в который будем собирать примеры изображений тестовой выборки
  y_test = [] # Создаем пустой список, в который будем собирать правильные ответы (метки классов: 0 - Феррари, 1 - Мерседес, 2 - Рено)
  print('Создание наборов данных для обучения модели...')
  for j, d in enumerate(sorted(os.listdir(путь))):
    files = sorted(os.listdir(путь + '/'+d))    
    count = len(files) * коэф_разделения
    for i in range(len(files)):
      sample = image.load_img(путь + '/' +d +'/'+files[i], target_size=(размер[0], размер[1])) # Загружаем картинку
      img_numpy = np.array(sample) # Преобразуем зображение в numpy-массив
      if i<count:
        x_train.append(img_numpy) # Добавляем в список x_train сформированные данные
        y_train.append(j) # Добавлеям в список y_train значение 0-го класса
      else:
        x_test.append(img_numpy) # Добавляем в список x_test сформированные данные
        y_test.append(j) # Добавлеям в список y_test значение 0-го класса
  display.clear_output(wait=True)
  print('Созданы выборки: ')
  x_train = np.array(x_train) # Преобразуем к numpy-массиву
  y_train = np.array(y_train) # Преобразуем к numpy-массиву
  x_test = np.array(x_test) # Преобразуем к numpy-массиву
  y_test = np.array(y_test) # Преобразуем к numpy-массиву
  x_train = x_train/255.
  x_test = x_test/255.
  print('Размер сформированного массива x_train:', x_train.shape)
  print('Размер сформированного массива y_train:', y_train.shape)
  print('Размер сформированного массива x_train:', x_test.shape)
  print('Размер сформированного массива y_train:', y_test.shape)
  return (x_train, y_train), (x_test, y_test)

def предобработка_данных(**kwargs):
  if kwargs['сетка'] == 'полносвязная':
    x_train = kwargs['изображения']/255. # Нормируем изображения, приводя каждое значение пикселя к диапазону 0..1
    x_train = x_train.reshape((-1, 28*28)) # Изменяем размер изображения, разворачивая в один вектор
    print('Размер сформированных данных:', x_train[0].shape) # Выводим размер исходного изображения
    return x_train
  elif kwargs['сетка'] == 'сверточная':
    x_train = kwargs['изображения']/255. # Нормируем изображения, приводя каждое значение пикселя к диапазону 0..1
    x_train = x_train.reshape((-1, 28,28,1)) # Изменяем размер изображения, разворачивая в один вектор
    print('Размер сформированных данных:', x_train[0].shape) # Выводим размер исходного изображения
    return x_train

def обработка_изображений(**kwargs):
  return сегментация.get_images(**kwargs)



















def распаковать_архив(откуда='', куда=''):
  proc = subprocess.Popen('unzip -q "' + откуда + '" -d ' + куда, shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
  proc.wait()



def показать_примеры_сегментации(**kwargs):
  сегментация.show_sample(**kwargs)




def загрузить_базу_ОДЕЖДА():
  (x_train_org, y_train_org), (x_test_org, y_test_org) = datasets.fashion_mnist.load_data() # Загружаем данные набора MNIST
  display.clear_output(wait=True)
  print('Данные загружены')
  print('Размер обучающей выборки:', x_train_org.shape) # Отобразим размер обучающей выборки
  print('Размер тестовой выборки:', x_test_org.shape) # Отобразим размер тестовой выборки
  return (x_train_org, y_train_org), (x_test_org, y_test_org)



def загрузить_базу_СТРОЙКА(**kwargs):
  if 'путь' in kwargs:
	  path = '/content/drive/MyDrive/' + kwargs['путь'] + 'AIU.zip'
  else:
	  path = '/content/drive/MyDrive/AIU.zip'
  print('Загрузка данных')
  print('Это может занять несколько минут...')
  распаковать_архив(
    откуда = '/content/drive/MyDrive/AIU.zip',
    куда = '/content'
  )
  распаковать_архив(
      откуда = "Notebooks.zip",
      куда = "/content"
  )
  x_train = np.load('xTrain_st.npy')
  x_test = np.load('xVal_st.npy')
  y_train = np.load('yTrain_st.npy')
  y_test = np.load('yVal_st.npy')  
  print('Загрузка данных (Готово)')
  return (x_train, y_train), (x_test, y_test)
  
  print('Загрузка данных...')
  urls = ['https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1WtUbopKzQw97W8DChDu0JidJYh08XQyy',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=14gsGpYv13IMUKXmjQEPhPt2bVpVkcJfY',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1A9IThR5f7dUIHgohDDJBeFyAZquTiYIL',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1PtMhqaPXYoJKuKjLy338rBgv-PsScYPh',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1qRAPeOgCZ0g9nikKmop4uWGEe9cCSm4B',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1MDJiPs1Lyh-ij5dldjAu-kwWWb2uPNKi',
            'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1S7bc5yR2fHsR81aDZsIpBcCfxf-43Cek'
            '']
  for url in urls:    
    output = 'data.zip' # Указываем имя файла, в который сохраняем файл
    gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
    if os.path.exists('data.zip'):
        break  
  # Скачиваем и распаковываем архив
  распаковать_архив(
      откуда = "data.zip",
      куда = "/content"
  )
  x_train = np.load('xTrain_st.npy')
  x_test = np.load('xVal_st.npy')
  y_train = np.load('yTrain_st.npy')
  y_test = np.load('yVal_st.npy')  
  print('Загрузка данных (Готово)')
  return (x_train, y_train), (x_test, y_test)
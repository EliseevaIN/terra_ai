import os, random, io, json

from tensorflow.keras.preprocessing.text import Tokenizer # Методы для работы с текстами и преобразования их в последовательности
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from gensim.models import word2vec # Импортируем gensim 

def показать_пример(количество):
    lst = os.listdir('Договоры/Договора432/')
    rnd = random.choice(lst)
    f = open('Договоры/Договора432/'+rnd, 'r', encoding='utf-8')
    for line in f:
        print(line)

def создать_выборки_договоров(
      договора,      
      xLen = 256, # Длина окна  по умолчанию
      step = 30, # Шаг  по умолчанию
      embeddingSize = 300): # Количество измерений для векторного пространства  по умолчанию
  agreements = договора[0]
  words = договора[1]
  wordsToTest = договора[2]
  # lower=True - приводим слова к нижнему регистру
  # char_level=False - просим токенайзер учитывать слова, а не отдельные символы
  global tags_index, clean_voc, xLen_test, step_test, embeddingSize_test, tokenizer

  xLen_test = xLen
  step_test = step
  embeddingSize_test = embeddingSize

  tokenizer = Tokenizer(lower=True, filters = '', char_level=False)

  tokenizer.fit_on_texts(words) # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
  clean_voc = {} 

  for item in tokenizer.word_index.items(): #Преобразуем полученный список 
    clean_voc[item[0]] = item[1] # В словарь, меняя местами элементы полученного кортежа 
  # Преобразовываем текст в последовательность индексов согласно частотному словарю
  tok_agreem = tokenizer.texts_to_sequences(words) # Обучающие тесты в индексы

  # Собираем список индексов и dummy encoded вектора
  def get01XSamples(tok_agreem, tags_index):
    tags01 = [] # Список для тегов
    indexes = [] # Здесь будут лежать индексы
  
    for agreement in tok_agreem: # Проходимся по каждому договору-списку
      tag_place = [0, 0, 0, 0, 0, 0] # Создаем вектор [0,0,0,0,0,0]
      for ex in agreement: # Проходимся по каждому слову договора
          if ex in tags_index: # Смотрим, если индекс оказался нашим тегом
            place = np.argwhere(tags_index==ex) # Записываем под каким местом лежит этот тег в своем списке
            if len(place)!=0: # Проверяем, чтобы тег действительно был
              if place[0][0]<6: # Первые шесть тегов в списке - открывающие
                tag_place[place[0][0]] = 1    # Поэтому ставим 1
              else: 
                tag_place[place[0][0] - 6] = 0  # Остальные теги закрывающие, так что меняем на ноль
          else:          
            tags01.append(tag_place.copy()) # Расширяем наш список с каждой итерацией. Получаем в конце длинный список из всех тегов в одном 
            indexes.append(ex) # Докидываем индекс слова в список индексов

    return indexes, tags01

  # Получение списка слов из индексов
  def reverseIndex(clean_voc, x):
    reverse_word_map = dict(map(reversed, clean_voc.items())) # Берем пары значений всего словаря и размечаем наоборот, т.е. value:key
    words = [reverse_word_map.get(letter) for letter in x] # Вытаскиваем по каждому ключу в список
    return words # Возвращаем полученный текст

  # Формируем выборку из индексов
  def getSetFromIndexes(wordIndexes, xLen, step): 
    xBatch = [] # Лист для фрагментов текста
    wordsLen = len(wordIndexes) # Определяем длинну текста
    index = 0 # Задаем стартовый индекс
    
    while (index + xLen <= wordsLen): # Пока сумма индекса с длинной фрагмента меньше или равна числу слов в выборке
      xBatch.append(wordIndexes[index:index+xLen]) # Добавляем X в лист фразментов текста
      index += step # Сдвигаемся на step

    return xBatch # Лист для фрагментов текста   

  # s1 Условия
  # s2 Запреты
  # s3 Стоимость, всё про цены и деньги
  # s4 Всё про сроки
  # s5 Неустойка
  # s6 Всё про адреса и геолокации
  tags_index = ['<s' + str(i) + '>' for i in range(1, 7)] # Получаем список открывающих тегов
  closetags = ['</s' + str(i) + '>' for i in range(1, 7)] # Получаем список закрывающих тегов
  tags_index.extend(closetags) # Объединяем все теги

  tags_index = np.array([clean_voc[i] for i in tags_index]) # Получаем из словаря частотности индексы всех тегов

  xData, yData = get01XSamples(tok_agreem,tags_index) # Распознаем теги и создаем список с ними, с индексами
  decoded_text = reverseIndex(clean_voc, xData) # Для создания списков с embedding-ами сначала преобразуем список индексов обратно в слова
    
 
    # Генерируем наборы с заданными параметрами окна
  xTrain = getSetFromIndexes(decoded_text, xLen, step) # Последовательность из xLen слов
  yTrain = getSetFromIndexes(yData, xLen, step) # Последовательность из xLen-тегов

    # Создаем выборки
  def getSets(model, senI, tagI):
    xVector = [] # Здесь будет лежать embedding представление каждого из индексов
    tmp = [] # Временный список
    for text in senI: # Проходимся по каждому тексту-списку
      tmp=[]
      for word in text: # Проходимся по каждому слову в тексте-списке
        tmp.append(model[word]) 

      xVector.append(tmp)

    return np.array(xVector), np.array(tagI)

      # Передаем в word2vec списки списков слов для обучения
  # size = embeddingSize - размер эмбеддинга
  # window = 10 - минимальное расстояние между словами в эмбеддинге 
  # min_count = 1 - игнорирование всех слов с частотой, меньше, чем 1
  # workers = 10 - число потоков обучения эмбеддинга
  # iter = 10 - число эпох обучения эмбеддинга

  modelGENSIM = word2vec.Word2Vec(xTrain, size = embeddingSize, window = 10, min_count = 1, workers = 10, iter = 10)

  xTrainGENSIM, yTrainGENSIM = getSets(modelGENSIM, xTrain, yTrain)
  tokenizer_json1 = tokenizer.to_json()
  with io.open('tokenizer_dog.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json1, ensure_ascii=False))
  
  with open('clean_voc.json', 'w') as fp:
    json.dump(clean_voc, fp)
    
  return xTrainGENSIM, yTrainGENSIM, tags_index   #, xLen, embeddingSize
  
def тест_модели(model, теги, договора):
  # получаем нужные параметры из функции формировавшие xtrain, yTrain
  xLen = 256
  step = 30
  embeddingSize = 300
  tags_index = теги
  with open('tokenizer_dog.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    
  with open('clean_voc.json') as json_file: 
    clean_voc = json.load(json_file) 
    
  print("Тест модели:")
  
  wordsToTest = договора[2]
 # Функция, выводящая точность распознавания каждой категории отдельно 
  def recognizeSet(tagI, pred, tags, length, value):
    total=0

    for j in range(6): # общее количество тегов
      correct=0
      for i in range(len(tagI)): # проходимся по каждому списку списка тегов
        for k in range(length): # проходимся по каждому тегу
          if tagI[i][k][j]==(pred[i][k][j]>value).astype(int): # если соответствующие индексы совпадают, значит сеть распознала верно
            correct+=1 
      print("Сеть распознала категорию '{}' на {}%".format(tags[j],round(100*correct/(len(tagI)*length), 2)))
      total += 100 * correct / (len(tagI)*length)
    print("Cредняя точность {}%".format(round(total/6,2)))

  def get01XSamples(tok_agreem, tags_index):
    tags01 = [] # Список для тегов
    indexes = [] # Здесь будут лежать индексы
  
    for agreement in tok_agreem: # Проходимся по каждому договору-списку
      tag_place = [0, 0, 0, 0, 0, 0] # Создаем вектор [0,0,0,0,0,0]
      for ex in agreement: # Проходимся по каждому слову договора
          if ex in tags_index: # Смотрим, если индекс оказался нашим тегом
            place = np.argwhere(tags_index==ex) # Записываем под каким местом лежит этот тег в своем списке
            if len(place)!=0: # Проверяем, чтобы тег действительно был
              if place[0][0]<6: # Первые шесть тегов в списке - открывающие
                tag_place[place[0][0]] = 1    # Поэтому ставим 1
              else: 
                tag_place[place[0][0] - 6] = 0  # Остальные теги закрывающие, так что меняем на ноль
          else:          
            tags01.append(tag_place.copy()) # Расширяем наш список с каждой итерацией. Получаем в конце длинный список из всех тегов в одном 
            indexes.append(ex) # Докидываем индекс слова в список индексов

    return indexes, tags01

  # Формируем выборку из индексов
  def getSetFromIndexes(wordIndexes, xLen, step): 
    xBatch = [] # Лист для фрагментов текста
    wordsLen = len(wordIndexes) # Определяем длинну текста
    index = 0 # Задаем стартовый индекс
    
    while (index + xLen <= wordsLen): # Пока сумма индекса с длинной фрагмента меньше или равна числу слов в выборке
      xBatch.append(wordIndexes[index:index+xLen]) # Добавляем X в лист фразментов текста
      index += step # Сдвигаемся на step

    return xBatch # Лист для фрагментов текста   

  # Получение списка слов из индексов
  def reverseIndex(clean_voc, x):
    reverse_word_map = dict(map(reversed, clean_voc.items())) # Берем пары значений всего словаря и размечаем наоборот, т.е. value:key
    words = [reverse_word_map.get(letter) for letter in x] # Вытаскиваем по каждому ключу в список
    return words # Возвращаем полученный текст

  # Создаем выборки
  def getSets(model, senI, tagI):
    xVector = [] # Здесь будет лежать embedding представление каждого из индексов
    tmp = [] # Временный список
    for text in senI: # Проходимся по каждому тексту-списку
      tmp=[]
      for word in text: # Проходимся по каждому слову в тексте-списке
        tmp.append(model[word]) 

      xVector.append(tmp)

    return np.array(xVector), np.array(tagI)

  # Преобразовываем текст в последовательность индексов согласно частотному словарю
  #tokenizer = Tokenizer(lower=True, filters = '', char_level=False)
  tok_agreemTest = tokenizer.texts_to_sequences(wordsToTest) # Обучающие тесты в индексы

  xDataTest, yDataTest = get01XSamples(tok_agreemTest,tags_index) # Распознаем теги и создаем список с ними, с индексами
  decoded_text = reverseIndex(clean_voc, xDataTest) # Для создания списков с embedding-ами сначала преобразуем список индексов обратно в слова  

  # Генерируем наборы с заданными параметрами окна
  xTest = getSetFromIndexes(decoded_text, xLen, step) # Последовательность из xLen слов
  yTest = getSetFromIndexes(yDataTest, xLen, step) # Последовательность из xLen-тегов

  # Передаем в word2vec списки списков слов для обучения
  # size = embeddingSize - размер эмбеддинга
  # window = 10 - расстояние между текущим и прогнозируемым словом в предложении
  # min_count = 1 - игнорирование всех слов с частотой, меньше, чем 1
  # workers = 10 - число потоков обучения эмбеддинга
  # iter = 10 - число эпох обучения эмбеддинга

  modelGENSIM = word2vec.Word2Vec(xTest, size = embeddingSize, window = 10, min_count = 1, workers = 10, iter = 10)  

  xTestGENSIM, yTestGENSIM = getSets(modelGENSIM, xTest, yTest)

  tags = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
  # s1 Условия
  # s2 Запреты
  # s3 Стоимость, всё про цены и деньги
  # s4 Всё про сроки
  # s5 Неустойка
  # s6 Всё про адреса и геолокации

  pred = model.predict(xTestGENSIM) # сделаем предсказание

  recognizeSet(yTestGENSIM, pred, tags, xLen, 0.9)

  pass
####################################################################################
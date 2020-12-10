import os, librosa
import numpy as np
import IPython.display as ipd # Воспроизведение звуковыйх файлов
from IPython import display
import librosa.display
from matplotlib import pyplot as plt
import time
from tqdm.notebook import tqdm_notebook as tqdm_
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib.patches import Rectangle

def показать_примеры_голосовых_команд():
  i_start = 0.5
  i_end = 2.5

  список_категорий = ['Кондиционер', 'Свет', 'Телевизор', 'Фон']
  директории = sorted(os.listdir('/content/Умный_дом/comands/'))
  for i in range(len(список_категорий)):
    print ('Пример голосовой команды «', список_категорий[i],'»:',sep='')
    список_примеров = os.listdir('/content/Умный_дом/comands/'+ директории[i])
    случайный_пример = список_примеров[np.random.randint(0, len(список_примеров[i]))]
    y,sr = librosa.load('/content/Умный_дом/comands/'+ директории[i]+'/'+случайный_пример)
    ipd.display(display.Audio(data=y, rate = sr))    
    print ('Спектр голосовой команды:')  
    X = librosa.stft(y[int(sr*i_start): int(i_end*sr)])
    Xdb = librosa.amplitude_to_db(abs(X))      
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.show()
    librosa.display.waveplot(y)
    plt.show()
    print()
    print()

def создать_выборки():
  время_старта = time.time()

  # все константы
  тренировочная_путь = "/content/Умный_дом/comands/" 
  length=11025
  sample_rate = 22050                   # Значение sample_rate аудиофайлов
  feature_dim_1 = 20                    # стандартная величина MFCC признаков
  feature_dim_2 = int(.5 * sample_rate) # установленная длина фреймов (в секундах 0.5 = 500 мс)
  step_mfcc = int(.04 * sample_rate)    # Шаг смещения при разборе mfcc (в секундах 0.1 = 100
  split_ratio=0.8                       # split_ratio - (1-split_ratio) равно доле тестовых образцов, которые вернет функция train_test_split
  random_state=42                       # random_state - начальное число, используемое генератором случайных чисел в функции train_test_split

  labels = sorted(os.listdir(тренировочная_путь)) # запишем лейблы классов по названию папок с данными - ['кондиционер', 'телевизор', 'свет', 'фон']  
  label_indices = np.arange(0, len(labels))       # запишем лейблы в виде индексов - [0, 1, 2, 3]

  def wav2mfcc(file_path, length = 11025, step = 2205): 
    out_mfcc = []                   # Выходной массив, содержащий mfcc исходного файла с шагом step
    out_audio = []                  # Выходной массив, содеражищий аудиоинформацию исходного файла с шагом step
    y, sr = librosa.load(file_path) # Загружаем данные исходного файла  

    while (len(y)>=length):                               # Проходим весь массив y, пока оставшийся кусочек не станет меньше указанной в параметре max_len длинны
      section = y[:length]                                # Берем начальный кусок длинной length
      section = np.array(section)                         # Переводим в numpy
      out_mfcc.append(librosa.feature.mfcc(section, sr))  # Добавляем в выходной массив out_mfcc значение mfcc текущего куска
      out_audio.append(section)                           # Добавляем в выходной массив аудио текущий кусок
      y = y[step:]                                        # Уменьшаем y на step
    
    out_mfcc = np.array(out_mfcc)     # Преобразуем в numpy
    out_audio = np.array (out_audio)  # Преобразуем в numpy
    return out_mfcc, out_audio        # функция вернет массив мэл-частот и массив аудио-отрезков

  '''
  Объявим функцию формирования и сохранения векторов данных, полученных для каждого набора аудио-команд в датасете
  Параметры:
    path  - путь к папке, в которой находятся каталоги с обучающими командами    
    length - длинна отрезков, на которые разбиваем исходный файл
  '''
  
  def save_data_to_array(path=тренировочная_путь, length=11025):
    for label in labels:    # для каждого лейбла
      mfcc_vectors = []     # здесь соберем векторы MFCC частот
      # извлечем для каждого файла его путь c названием папки и именем файла и соберём в список
      # ['/content/drive/My Drive/Речевые команды/свет/recording12.wav', '/content/drive/My Drive/Речевые команды/свет/recording11.wav'..]
      wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
      for wavfile in tqdm_(wavfiles, desc=('Сохраняем векторы класса ' + label), ncols=1000):
        mfcc, _ = wav2mfcc(wavfile, length=length, step = step_mfcc)                  # получим мел-частоты      
        if (mfcc.shape[0] != 0 ):                                                     # Если массив не нулевой длинны
          mfcc_vectors.extend(mfcc)                                                   # и добавим вектор в список для соответствующего класса
      np.save(label + '.npy', mfcc_vectors)                                           # сохраним массивы данных для каждого класса
  
  save_data_to_array(length=feature_dim_2)
   
  # подготовка выборок
  # откусываем первый кусок
  X = np.load('/content/'+labels[0] + '.npy')   # берем набор векторов для первого класса 
  y = np.zeros(X.shape[0], dtype = 'int32')     # устанавливаем размер соответствующего ему лейбла

  # Объединяем в единый датасет в виде np-массива обучающий и проверочный набор данных
  for i, label in enumerate(labels[1:]): 
    x = np.load('/content/'+label + '.npy')                     # Читаем очередной массив данных 
    X = np.vstack((X, x))                                       # Соединяем с исходным набором
    y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))  # В массив y добавлем x.shape[0]-элеменентов со значением (i + 1)

  # Формируем обучающую и проверочную выборки
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= (1-split_ratio), random_state=random_state, shuffle=True)
  
  # Добавляем значение канала(=1) в размеры 'x' выборок
  x_train = x_train[..., None]
  x_test = x_test[..., None]

  y_train = to_categorical(y_train)   # представляем лейблы классов обучающего набора в виде one-hot вектора ((0, 1, 0) и т.п)
  y_test = to_categorical(y_test)     # представляем лейблы классов проверочного набора в виде one-hot вектора ((0, 1, 0) и т.п)

  число_аудио = 0
  for i in labels:
    число_аудио += len(os.listdir(тренировочная_путь+i))
  print()
  print('Данные успешно загружены. Обработано: ' + str(число_аудио) + ' аудио')
  print()
  время_на_обработку = time.time() - время_старта
  print('Время обработки: '+ str(int(время_на_обработку//60)) + ' мин '+ str(round(время_на_обработку%60,2)) +' c')
  return (x_train, y_train), (x_test, y_test)

def тест_модели(модель, порог):
  sample_rate = 22050                   # Значение sample_rate аудиофайлов
  feature_dim_2 = int(.5 * sample_rate) # установленная длина фреймов (в секундах 0.5 = 500 мс)
  step_mfcc = int(.02 * sample_rate)    # Шаг смещения при разборе mfcc (в секундах 0.1 = 100
  classes = ['КОНДИЦИОНЕР', 'СВЕТ', 'ТЕЛЕВИЗОР']
  n_classes = 4

  #  функция параметризации аудио(wav в мел-частоты)
  def wav2mfcc(file_path, length = 11025, step = 2205): 
    out_mfcc = []                   # Выходной массив, содержащий mfcc исходного файла с шагом step
    out_audio = []                  # Выходной массив, содеражищий аудиоинформацию исходного файла с шагом step
    y, sr = librosa.load(file_path) # Загружаем данные исходного файла  

    while (len(y)>=length):                               # Проходим весь массив y, пока оставшийся кусочек не станет меньше указанной в параметре max_len длинны
      section = y[:length]                                # Берем начальный кусок длинной length
      section = np.array(section)                         # Переводим в numpy
      out_mfcc.append(librosa.feature.mfcc(section, sr))  # Добавляем в выходной массив out_mfcc значение mfcc текущего куска
      out_audio.append(section)                           # Добавляем в выходной массив аудио текущий кусок
      y = y[step:]                                        # Уменьшаем y на step
    
    out_mfcc = np.array(out_mfcc)       # Преобразуем в numpy
    out_audio = np.array (out_audio)    # Преобразуем в numpy
    return out_mfcc, out_audio          # функция вернет массив мэл-частот и массив аудио-отрезков

  # Объявим функцию предсказания команды
  def predict(namefile, model, min_count = 2, rate = порог, hole = 1):                    # функция принимает на вход путь к нужному файлу, и имя обученной модели 
    mfcc_full, audio_full = wav2mfcc(namefile, length=feature_dim_2, step = step_mfcc)  # Получаем массив mfcc выбранного файла с именем namefile    

    #mfcc = xScaler.transform(mfcc_full.reshape(-1,1))
    mfcc_full = mfcc_full.reshape(-1, 20, 22, 1)
    g_pred = model.predict(mfcc_full)               # Предиктим с помощью модели model массив mfcc
    pred = np.array([np.argmax(i) for i in g_pred]) # Выбираем индекс максимального элемента в каждом g_pred[i]  и создаем numpy-массив pred из этих индексов

    out = []    # Объявляем выходную переменную out (В ней будут храниться преобразованные из mfcc ауидоданные, класс команды и точность, с которой сеть считает эту команду верной)  
  
    # Ищем команды каждого класса
    for idx_class in range(n_classes-1):
      idxs = np.where(pred == idx_class)  # В массиве pred находим все элементы со значением, равным искомому классу idx_class
      idxs = idxs[0]                      # Размерность полученного маасива в np.where иммет тип (x,). Оставляем только первую размерность
      if (len(idxs) == 0):                # Если элементы искомого класса не найдены,
        continue                          # то переходим к поиску команд следующего класса

      curr = [] # Временный массив для хранения информации о найденных командах
      '''
      в массиве idx данные прдеставлены следующим образом:
      [4, 5, 6, 7, 123, 124, 125, 126, 127]
      в массив curr мы запишем [4, 123] только стартовые индексы
      поскольку очевидно, что 4,5,6,7 и 123,124,125,126,127 представляют единую команду
      '''
      curr_idx =int(idxs[0])  # Текущий стартовый индекс
      summ, length = 0, 0     # summ - хранит сумму вероятностей, с которой сеть отнесла команду к данному классу; length - длинна последовательно идущих элементов для одной команды (для массива curr из примера                                                                        
                              #[4, 123] длина соответствующая первому элементу будет 4, второму - 5 )
      
      for i in range(len(idxs)):              # Пробегаем по всему массиву idxs
        summ += g_pred[idxs[i]][idx_class]    # Считаем сумму вероятности
        length += 1                           # Увеличиваем длинну последовательности
        if i == len(idxs)-1:                  # Если последний элемент последовательности
          if (length >= min_count and summ / length >= rate):   # Проверяем на условия разбора: длинна последовательности должна быть больше входного параметра min_count
                                                                # summ / length должно быть больше входного параметра rate
            curr.append([curr_idx, length, summ / length])      # Если условия выполняются, то добавляем в маасив стартовый индекс найденной команды, длинну последовательности и summ / length

          break  
        if idxs[i+1]-idxs[i]>hole:                            # Если следующий индекс больше текущего на hole (означает, что следующий элемент относится уже к другой комманде)
          if (length >= min_count and summ / length >= rate): # Проверяем на условия разбора: длинна последовательности должна быть больше входного параметра min_count
                                                              # summ / length должно быть больше входного параметра rate
 #           print(length)
            curr.append([curr_idx, length, summ / length])  # Если условия выполняются, то добавляем в маасив стартовый индекс найденной команды, длинну последовательности и summ / length
          curr_idx = int (idxs[i+1])                        # Изменяем текущий стартовый индекс
          summ, length = 0, 0                               # Обнуляем summ и length
      curr_audio = [] 
      for elem in curr:                                     # Проходим по всему массиву curr
        curr_audio = audio_full[elem[0]]      # Если это стартовый элемент исходных данных, то берем самую первую mfcc
        for j in range(1,elem[1]):            # Пробегаем цикл от 1 до elem[1]+1 (где elem[1] хранит длинну последовательности элементов, отнесенных к одной команде)
         if (elem[0]+j == len(audio_full)):   # Если elem[0] + j равно длинне mfcc, то выходим из цикла
           break
         curr_audio = np.hstack((curr_audio, audio_full[elem[0]+j][-step_mfcc:]))
        curr_audio = np.array(curr_audio)             # Переводим массив в numpy
        out.append([curr_audio, idx_class, elem])  # Добавляем данные в выходной массив
    return out, pred                         # Возращаем массив с данными, массив с классами команд, массив с softmax данными

  проверочная_выборка = '/content/Умный_дом/test_speech/'
  список_проверочной = os.listdir(проверочная_выборка)
  wavfiles =  проверочная_выборка+список_проверочной[np.random.randint(len(список_проверочной))]  # Получаем имя случайного файла
  print ('Исходное аудио для проверки работы нейронной сети: ')
  y_org,sr = librosa.load(wavfiles)      
  display.display(display.Audio (y_org, rate = 22050))
  print()

  out, pred = predict(wavfiles, model=модель, min_count = 8, rate = порог, hole = 1)                # Вызываем predict для очередного файла  
  color = ['y','g','b']
  if (len(out)==0):                                                                                   # Если длинна массива равна 0, то команда не распознана
    print('Команда не распознанана!!!')
    return
  for elem in out[:1]: 
    print ('Распознана команда: "', classes[elem[1]], '" (вероятность - %.2f' % (elem[2][2]*100), '%)')  # Выводим название
    display.display(display.Audio(elem[0], rate = 22050))
    fig = plt.figure(figsize=(30,5))
    ax = fig.add_subplot(111) 
    plt.plot(y_org)
    ax.add_patch(Rectangle((441*elem[2][0], 1.), 
                            20000, -2, 
                              fc =color[elem[1]],  
                              ec =color[elem[1]], 
                              fill=True,
                              lw = 2,
                              alpha=0.5) ) 
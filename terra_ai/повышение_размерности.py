from . import датасет, модель
from os import listdir
from PIL import Image as PImage
import tensorflow as tf #Импортируем tensorflow
import matplotlib.pyplot as plt
from PIL import Image
import random
import importlib.util, sys, gdown
from IPython import display

import tensorflow as tf #Импортируем tensorflow
from tensorflow.python.data.experimental import AUTOTUNE #Импортируем AUTOTUNE для создания нескольких процессорных потоков при обучении 
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19 #Импортируем метод загрузки датасетов из директории во время обучения сети
#VGG19 - метод создания продвинутой модели нейронной сети для работы с полноразмерными изображениями 

from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError #Импортируем методы подсчета ошибок
from tensorflow.keras.metrics import Mean #Импортируем метрику
from tensorflow.keras.optimizers import Adam #Импортируем оптимайзеры
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, MaxPooling2D #Импортируем слои
from tensorflow.python.keras.models import Model, Sequential  #Импортируем метод создания модели

import os #Импортируем для работы с файловой системой
import matplotlib.pyplot as plt #Импортируем для вывода изображений
import numpy as np #Импортируем для работы с матрицами
from PIL import Image #Импортируем для загрузки изображений из директории
import time #Импортируем для подсчета времени 

def генератор_данных_DIV2K():
  curr_time = time.time()
  print('Загрузка и обработка данных')
  print('Это может занять несколько минут...') 
  print() 
  #Класс, созданный разработчиками архитектуры SRGAN
  #Нужен для создания выборки из директории с фотографиями прямо во время обучения 
  class DIV2K:
      def __init__(self,
                  subset='train', #Определяем выборку при создании 
                  images_dir='.div2k/images', #Директория для хранения тестовой и обучающей выборки
                  caches_dir='.div2k/caches'): #Директория для хранения кэша 

          self.scale =  4 #Масштаб уменьшения изображения по каждой оси  (суммарно изображение, подаваемое на вход меньше возвращаемого в 4 раза)
          self.downgrade = 'bicubic' #Определяем метод с которым уменьшаются изображения

          #Все фотографии в датасете пронумерованы по порядку. Поэтому имена для 
          #загрузки будут формироваться по порядку при помощи подствления к порядковому номеру 
          #формата изображения (.png). Ниже в зависимости от выборки определяем какие порядковые номера
          #изображений нам нужно брать, если на вход функции было подано train или valid.
          #Иначе выдаем ошибку
          if subset == 'train':
              self.image_ids = range(1, 801)
          elif subset == 'valid':
              self.image_ids = range(801, 901)
          else:
              raise ValueError("subset must be 'train' or 'valid'")

          self.subset = subset #Записываем тип выборки 
          self.images_dir = images_dir #Записываем путь для изображений
          self.caches_dir = caches_dir #Записываем путь для кэша изображений

          #Создаем папки для обучающей и тестовой выборки 
          os.makedirs(images_dir, exist_ok=True)
          os.makedirs(caches_dir, exist_ok=True)

      def __len__(self):
          return len(self.image_ids)

      #Функция для создания датасета. На вход получает:
      #batch_size - размер батча 
      #repeat_count - число повторений датасета 
      #random_transform - делаем ли трансофрмации изображения
      def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
          #Формируем датасет, создавая струткуру [(изображение1 в плохом разрешении, изображение1 в хорошем разрешении), (изображение2 в плохом разрешении, изображение2 в хорошем разрешении)]
          ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
          if random_transform: #Если нужно делать трансформации
              ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE) #Делаем случайную обрезку изображений
              ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE) #Случайно поворачиваем изображения
              ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE) #Случайно зеркалим изображения
          ds = ds.batch(batch_size) #Разбиваем на батчи
          ds = ds.repeat(repeat_count) #Делаем повторения, если требуется 
          #Делаем многопоточную загрузку данных 
          #Создаем набор данных. Большинство конвейеров ввода набора данных должны заканчиваться вызовом предварительной выборки. 
          #Это позволяет подготовить более поздние элементы, пока обрабатывается текущий элемент. 
          #Это часто уменьшает задержку и увеличивает пропускную способность за счет использования дополнительной памяти для хранения предварительно выбранных элементов.
          ds = ds.prefetch(buffer_size=AUTOTUNE)
          return ds #Возвращаем датасет

      #Функция для создания датасета из фотографий с высоким разрешением 
      def hr_dataset(self):
          
          if not os.path.exists(self._hr_images_dir()):#Если не в директории нет фотографий
              download_archive(self._hr_images_archive(), self.images_dir, extract=True) #Скачиваем их 

          ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file()) #Формируем датасет, создавая кэш фотографий 
          #Кэш нужен для ускорения загрузки данных 

          if not os.path.exists(self._hr_cache_index()): #Если не в директории нет кэша
              self._populate_cache(ds, self._hr_cache_file()) #Создаем кэш

          return ds #Возвращаем датасет

      #Функция для создания датасета из фотографий с низким разрешением 
      def lr_dataset(self):
          if not os.path.exists(self._lr_images_dir()): #Если не в директории нет фотографий
              download_archive(self._lr_images_archive(), self.images_dir, extract=True) #Скачиваем их 

          ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file()) #Формируем датасет, создавая кэш фотографий 
          #Кэш нужен для ускорения загрузки данных 

          if not os.path.exists(self._lr_cache_index()): #Если не в директории нет кэша 
              self._populate_cache(ds, self._lr_cache_file()) #Создаем кэш

          return ds

      def _hr_cache_file(self): #Формируем путь кэша изображений с высоким качеством
          return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_HR.cache')

      def _lr_cache_file(self): #Формируем путь кэша изображений с низким качеством
          return os.path.join(self.caches_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.cache')

      def _hr_cache_index(self): #Формируем индексы кэша изображений с высоким качеством
          return f'{self._hr_cache_file()}.index'

      def _lr_cache_index(self): #Формируем индексы кэша изображений с низким качеством
          return f'{self._lr_cache_file()}.index'

      def _hr_image_files(self): #Функция для создания списка путей изображений с высоким качеством для датасета
          images_dir = self._hr_images_dir() #Задаем путь 

          #Возвращаем список изображений с высоким качеством для датасета
          return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids] 

      def _lr_image_files(self): #Функция для создания списка путей изображений с низким качеством для датасета
          images_dir = self._lr_images_dir() #Задаем путь

          #Возвращаем список изображений с низким качеством для датасета
          return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

      #Формируем имена для изображений по id
      def _lr_image_file(self, image_id):
              return f'{image_id:04}x{self.scale}.png'

      #Формируем путь к изображениям высокого качества
      def _hr_images_dir(self):
          return os.path.join(self.images_dir, f'DIV2K_{self.subset}_HR')

      def _lr_images_dir(self):
          return os.path.join(self.images_dir, f'DIV2K_{self.subset}_LR_{self.downgrade}', f'X{self.scale}')

      #Формируем путь архиву с изображениями высокого качества
      def _hr_images_archive(self):
          return f'DIV2K_{self.subset}_HR.zip'

      #Формируем путь к архиву с изображениями низкого качества
      def _lr_images_archive(self):
          return f'DIV2K_{self.subset}_LR_{self.downgrade}_X{self.scale}.zip'

      #Функция для формирования датасета
      @staticmethod
      def _images_dataset(image_files): #Передаем пути изображений
          ds = tf.data.Dataset.from_tensor_slices(image_files) #Преобразуем в датасет
          ds = ds.map(tf.io.read_file) #По каждому пути читаем изображение и вносим в датасет 
          ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE) #Преобразуем четырехканальные изображения в трехканальные 
          return ds #Возвращаем датасет 

      #Функция для вывода информации о формировании кэша
      @staticmethod
      def _populate_cache(ds, cache_file):
          pass
          #print(f'Caching decoded images in {cache_file} ...')
          #for _ in ds: pass
          #print(f'Cached decoded images in {cache_file}.')


  #Функция для случайной вырезки фразментов из изображений с низким качеством и высоким качеством 
  #Принимает на вход параметры:
  #lr_img - изображение с низким качеством
  #hr_img - изображение с высоким качеством 
  #hr_crop_size - размер обрезанного изображения с выоским качеством
  #scale - масштаб во сколько раз по каждой из осей изображение с низким качеством меньше изображения с выоским качеством
  def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
      lr_crop_size = hr_crop_size // scale #Считаем размер обрезанного изображения с низким качеством
      lr_img_shape = tf.shape(lr_img)[:2] #Тензорная размерность уменьшенного изображения

      #Задаем случайные координаты для фрагментов, которые будем вырезать из фотографии
      lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
      lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

      #Умножаем координаты на масштаб, чтобы на фотографиях с большим и маленьким разрешением вырезать соответствующие фрагменты 
      hr_w = lr_w * scale
      hr_h = lr_h * scale

      #Вырезаем фрагменты
      lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
      hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

      return lr_img_cropped, hr_img_cropped # Возвращаем фрагменты 


  #Функция для отзеркаливания изображений
  def random_flip(lr_img, hr_img):
      #Случайно определяем стоит ли зеркалить изображения или нет 
      rn = tf.random.uniform(shape=(), maxval=1)
      #Возвращаем изображения 
      return tf.cond(rn < 0.5,
                    lambda: (lr_img, hr_img),
                    lambda: (tf.image.flip_left_right(lr_img),
                              tf.image.flip_left_right(hr_img)))

  #Функция для поворота изображений
  def random_rotate(lr_img, hr_img):
      #Случайно определяем стоит ли поворачивать изображения или нет 
      rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
      #Возвращаем изображения 
      return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

  #Функция для скачивания архива
  def download_archive(file, target_dir, extract=True):
      source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
      target_dir = os.path.abspath(target_dir)
      tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
      os.remove(os.path.join(target_dir, file))

  div2k_train = DIV2K(subset='train') #Инициализируем тренировочный датасет 
  div2k_valid = DIV2K(subset='valid') #Инициализируем тестовый датасет 
  train_ds = div2k_train.dataset(batch_size=16, random_transform=True) #Инициализируем тренировочный датасет 
  valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1) #Инициализируем тестовый датасет 
  display.clear_output(wait=True)
  print('Загрузка данных завершена! Длительность загрузки:', round(time.time() - curr_time, 2), 'секунд')  
  return train_ds, valid_ds

def загрузить_веса_готовой_модели():
  # Location of model weights (needed for demo)

  
  url = 'https://storage.googleapis.com/aiu_bucket/gan_generator.h5'
  output = 'gan_generator.h5' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL
  
  gan_generator = модель.создать_модель_HighResolution()()

  gan_generator.load_weights('gan_generator.h5')
  return gan_generator



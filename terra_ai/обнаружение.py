import random, os, operator, cv2, shutil, gdown,sys, time
import matplotlib.pyplot as plt
import albumentations as A
import subprocess
from subprocess import STDOUT, check_call
from tqdm.notebook import tqdm
from IPython import display
from PIL import Image

def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image"""
    '''
    Note: The coco format of a bounding box looks like [x_min, y_min, width, height], e.g. [97, 12, 150, 200]. 
    The pascal_voc format of a bounding box looks like [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212]. 
    The yolo format of a bounding box looks like [x, y, width, height], e.g. [0.3, 0.1, 0.05, 0.07].

    '''
    '''
    Calculations for YOLO
    '''
    x, y, w, h, _ = bbox
    boxXY = (x, y)
    halfboxWH = (w//2, h//2)
     
    topLeft = tuple(map(operator.sub, boxXY, halfboxWH))   # top left
    bottomRight = tuple(map(operator.add, boxXY, halfboxWH))    # bottom right

    x_min, y_min = int(round(topLeft[0])), int(round(topLeft[1]))
    x_max, y_max = int(round(bottomRight[0])), int(round(bottomRight[1]))

    # print (x_min, y_min)
    # print (x_max, y_max)
    '''
    Calculations for COCO 
    x_min, y_min, w, h, _ = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    '''

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img

def get_bboxed_img(image, bboxes, bbox_класс_номер, category_id_to_name):
    img = image.copy()
    class_labels = list(category_id_to_name.values())
    for bbox, bbox_класс in zip(bboxes, bbox_класс_номер):
        class_name = category_id_to_name[bbox_класс]
        # bbox = A.augmentations.bbox_utils.convert_bbox_to_albumentations (bbox, 'yolo', img.shape[0], img.shape[1], check_validity=True)
        bbox = A.augmentations.bbox_utils.denormalize_bbox(bbox, img.shape[0], img.shape[1])
        # print(bbox)
        img = visualize_bbox(img, bbox, class_name)
    return img


def взять_bbox_данные(путь_файл, img_width, img_height):
    bbox_лист = []
    bbox_класс_номер = []
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']

    valid_classes = [0]
    category_id_to_name = {i:f'{class_names[i]}' for i in range(len(class_names))}
    
    # bbox_коорд = []
    with open(путь_файл, mode='r', encoding='utf-8') as txtfile:
        for line in txtfile:
            bbox_data = line.split()
            bbox_коорд = []
            if int(bbox_data[0]) in valid_classes:
                for i in range(1,len(bbox_data)):
                  bbox_коорд.append(float(bbox_data[i]))
                bbox_класс_номер.append(int(bbox_data[0]))
                bbox_коорд.append(category_id_to_name[int(bbox_data[0])])
                bbox_лист.append(bbox_коорд)
    return bbox_лист, bbox_класс_номер


def взять_пары_картинки_лейблы(имя_семпла, путь_картинки, путь_лейблы):
    img = plt.imread(os.path.join(путь_картинки, f'{имя_семпла}.jpg'))
    # print(img.shape)
    bbox_лист, bbox_класс_номер = взять_bbox_данные(os.path.join(путь_лейблы, 
                                                                 f'{имя_семпла}.txt'), 
                                                    img.shape[0], 
                                                    img.shape[1]) 
    return img, bbox_лист, bbox_класс_номер


def show_pairs(display_list, figsize=(14,28)):
    plt.figure(figsize=figsize)
    строк = int(round(len(display_list)/2+0.1, 0))
    title = ['Оригинальное изображение', 'Размеченное изображение']
    for ix in range(строк*2):
        plt.subplot(строк, 2, ix+1)
        plt.title(title[int(ix%2)], fontsize=20)
        plt.imshow(display_list[ix])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def показать_примеры():
    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White
    samples_pool = ['000000007816', '000000021903', 
                    '000000005060', '000000002532',
                    '000000009483', '000000038829']

    samples_qty = 4
    имена_семплов = random.sample(samples_pool, samples_qty)
    class_names = ['', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']

    valid_classes = [0]
    category_id_to_name = {i:f'{class_names[i]}' for i in range(len(class_names))}
    путь_картинки = '/content/coco/images/val2017/'
    путь_лейблы = '/content/coco/labels/val2017/'
    путь_детект = '/content/yolov5/data/images'
    display_list = [] 
    for индекс, имя_семпла in enumerate(имена_семплов):
      img, bbox_лист, bbox_class_number = взять_пары_картинки_лейблы(имя_семпла, путь_картинки, путь_лейблы)
      bboxes = bbox_лист
      transform = A.Compose([A.CenterCrop(400,400),], 
                            bbox_params=A.BboxParams(format='yolo', 
                                                     min_visibility=0.2, 
                                                    #  label_fields=bbox_class_number
                                                    )
                            )
      transformed = transform(image=img, bboxes=bboxes)
      # transformed = transform(image=img, bboxes=bboxes, class_labels="bbox_class_number")
      transformed_image = transformed['image']
      transformed_bboxes = transformed['bboxes']
      # transformed_class_labels = transformed['bbox_class_number']
      bboxed_img = get_bboxed_img(transformed_image, transformed_bboxes, bbox_class_number, category_id_to_name)
      display_list.append(transformed_image)
      display_list.append(bboxed_img)
    show_pairs(display_list)
    pass

def cоздать_модель_YOLO():
  print('Создание модели')  
  url = 'https://storage.googleapis.com/aiu_bucket/my_train.py'          
  output = '/content/yolov5-master/my_train.py' # Указываем имя файла, в который сохраняем файл
  gdown.download(url, output, quiet=True) # Скачиваем файл по указанному URL 
  print('Загрузка весов обученной модели')
  print('Это может занять несколько минут...')
  выполнить_команду('python /content/yolov5-master/train.py --img 640 --batch 16 --epochs 0 --data coco128.yaml --weights yolov5s.pt --device 0')
  display.clear_output(wait=True)
  выполнить_команду('mkdir тест_обнаружение_людей')
  print('Загрузка модели завершена')
    
  
'''
Выводим результаты теста модели
'''
def show_detection_results(sample_name):
    print(sample_name)
    путь_картинки = '/content/coco/images/val2017/'
    путь_лейблы = '/content/coco/labels/val2017/'
    путь_детект = '/content/тест_обнаружение_людей'
    detected_img = plt.imread(os.path.join(путь_детект, f'{sample_name}'))

    transform = A.Compose([A.CenterCrop(detected_img.shape[0], detected_img.shape[1]),], 
                            bbox_params=A.BboxParams(format='yolo', 
                                                     min_visibility=0.2, 
                                                    #  label_fields=bbox_class_number
                                                    )
                            )
    
    img, bboxes, bbox_class_number = взять_пары_картинки_лейблы(sample_name, путь_картинки, путь_лейблы)
    transformed = transform(image=img, bboxes=bboxes)
    # transformed = transform(image=img, bboxes=bboxes, class_labels="bbox_class_number")
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    category_id_to_name = {0: ''}
    bboxed_img = get_bboxed_img(transformed_image, transformed_bboxes, bbox_class_number, category_id_to_name)

    detected_transformed = transform(image=detected_img, bboxes=bboxes)
    transformed_detected_img = detected_transformed['image']
    transformed_detected_bboxes = detected_transformed['bboxes']
    # bboxed_detected = get_bboxed_img(transformed_detected_img, transformed_detected_bboxes, bbox_class_number, category_id_to_name)

    # display_list = [transformed_image, bboxed_img, transformed_detected_img, bboxed_detected ]
    display_list = [transformed_image, transformed_detected_img]
    title = ['Оригинальное изображение', 'Результат обнаружения']
    plt.figure(figsize=(16, 20))
    for ix in range(2):
        plt.subplot(1, 2, ix+1)
        plt.title(title[ix], fontsize=20)
        plt.imshow(display_list[ix])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    pass


def выполнить_команду(команда='!ls'):
  proc = subprocess.Popen(f'{команда}', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
  proc.wait()
  pass

def тест_модели(NN_Chkpt_Name='yolov5m', treshhold=0.3):
  fname = files.upload()
  fname = list(fname.keys())[0]

  путь_детект = '/content/тест_обнаружение_людей/'
  
  выполнить_команду(f'cp /content/{fname} {путь_детект}')
  выполнить_команду('python /content/yolov5-master/detect.py --weights yolov5m.pt --img 640 --conf 0.3 --class 0 --source /content/тест_обнаружение_людей')
  path = '/content/runs/detect/'
  num = sorted(os.listdir('/content/runs/detect/'))[-1]
  path+=str(num)
  im = Image.open(path+'/'+fname)
  plt.figure(figsize = (14,7))
  plt.imshow(im)
  plt.axis('off')
  plt.show()

  
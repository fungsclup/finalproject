import matplotlib.pyplot as plt
import numpy as np
import tflearn
import os
import cv2
import sys
import tkinter
from tkinter import filedialog

# network için gerekli olan atamaları yaptım. Fonksiyonun kodlamalarına tflearn/layers'tan ulaşılabilir.
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
#değişkenleri burda da görmek açısından yeniden tanımladım.
LR = 0.001
IMG_SIZE = 64
MODEL_NAME = './data/dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
test_data = np.load('test_data.npy') #şimdi test edeceğimiz resim dosyalarının datasını çekiyoruz.

#network ataması
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
###########################################################


#networkü modelimize bağladık.
model = tflearn.DNN(convnet, tensorboard_dir='log')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
fig = plt.figure()

# python test.py şeklinde çalıştırıldığında default olarak bu çalışacak toplamda 12 resimi test ediyoruz.
def more_than_one():
    plt.gray()
    for num, data in enumerate(test_data[24:36]):
        # tensorboard --logdir=log/ burdaki komut log dosyası içerisindeki verileri tensorboard üzerinden görmemizi yazıyor.
        #farklı bir dizinden çağırmak için logdir'den sonra log klasörünün tamamını yazmak lazım
        #örneğin tensorboard --logdir=C:/task3/log/ gibi buranın çıktığısını zaten ekte size gönderdim.
        img_num = data[1]
        img_data = data[0]
        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0] # burda asıl işlemi yapan fonksiyon budur. -- model.predict bu fonksiyonu input ile besledikten sonra
        #bize sonucu yazdırıyor.
        print(np.argmax(model_out)) #bizde bu dizide ortaya çıkan maksimum değerin yerini yazdırıyoruz numpy ile [0,0,0] şeklinde olan 1x3 array
        print(model_out) #sayısal değerleri görmek için predict sonucunuda gösteriyorum
        #################### En büyük sayı hangi kolondan çıkıyorsa onu title olarak atıyorum. ################
        if np.argmax(model_out) == 0:
            str_label = 'köpek'
        elif np.argmax(model_out) == 1:
            str_label = 'kedi'
        elif np.argmax(model_out) == 2:
            str_label = "çiçek"
        ########################################################################################################
        y.imshow(orig) # fotoğrafın işlenmiş hali değilde orjinal halini gösteriyorum
        plt.title(str_label) # title'ı yazdırıyorum
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


def testing_one_file():
  test_file = filedialog.askopenfilename()
  plt.gray()
  np_test_data = []
  test_file_data = cv2.imread(test_file,cv2.IMREAD_GRAYSCALE)
  print(test_file_data)
  test_file_data = cv2.resize(test_file_data, (IMG_SIZE, IMG_SIZE))
  np_test_data = [np.array(test_file_data), '5']
  img_num = np_test_data[1]
  img_data = np_test_data[0]
  orig = img_data
  data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
  plt.imshow(orig)
  model_out = model.predict([data])[0]
  print(np.argmax(model_out))
  print(model_out)
  if np.argmax(model_out) == 0:
    str_label = 'köpek'
  elif np.argmax(model_out) == 1:
    str_label = 'kedi'
  elif np.argmax(model_out) == 2:
    str_label = "çiçek"
  plt.imshow(orig)
  plt.title(str_label)
  plt.show()

if len(sys.argv) > 1:
    testing_one_file()
else:
    more_than_one()

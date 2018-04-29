import matplotlib.pyplot as plt
import numpy as np
import tflearn
import os
import cv2
import sys
from tkinter import filedialog
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import random
import datetime

LR = 0.001
MODEL_NAME = './data/notalar-{}-{}.model'.format(LR, '2conv-basic')
numpy_dataset_test = "dataset_test.npy"
numpy_labels = "labels_with_name.npy"
IMG_SIZE = 64

test_data = np.load(numpy_dataset_test)
random.shuffle(test_data)

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
convnet = fully_connected(convnet, 25, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
fig = plt.figure()
def labelling(index):
  label = np.load(numpy_labels)
  return label[index][1]
def more_than_one():
    plt.gray()
    for num, data in enumerate(test_data[24:36]):
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
        ########################################################################################################
        y.imshow(orig) # fotoğrafın işlenmiş hali değilde orjinal halini gösteriyorum
        plt.title(labelling(np.argmax(model_out))) # title'ı yazdırıyorum
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
########################################################################################################################
#
#
#
#   Size gönderdiğim önceki kodlamadan tek farkı bundan sonra command prompt üzerinden yapılan işlemlerde 
#   test.py folder=KlasorIsmi şeklinde başladığında bu alttaki fonksiyon çalışmaya başlayacak.
#	Bu işlemde autocrop fonksiyonu autocrop(Dosyaismi) şeklinde çağırıyor.
#
########################################################################################################################
def autocrop(klasor_adi): #klasörün adını giriş argümanından çekiyorum
  dosyaisimleri_ve_tahminler = []
  dosyalar = []	# dosya isimleri
  image_datas = [] # resimlerin okunduktan sonraki dijital verilerini burda saklıyorum
  img_pred = [] # Tahmin sonuçları
  if (klasor_adi != "" and os.path.exists(klasor_adi)): # fonksiyonun doğru çağırılması için işlem yapıyorum
    """  BU METHODU KULLANMIYORUM ARTIK SIRALI YAZDIRMIYOR
    for dosyaismi in os.listdir(klasor_adi): # döngü başlatıyorum ve klasordeki dosyaları sıralıyorum
      if (dosyaismi.split(".")[1] == "jpg"): #sonu .jpg ile bitenleri yani image dosyalarını kontrol edip
        dosyalar.append(dosyaismi) # diziye aktarıyorum
    """

    # DOSYALARI SIRALI ÇAĞIRMASI İÇİN ÖNCEKİ KODLAMAYI DEĞİŞTİRDİM DİĞER TÜRLÜ SIRAYLA ÇAĞIRMIYOR.
    dosya_sayisi = len(os.listdir(klasor_adi))
    i = 0
    for i in range(dosya_sayisi):
      dosya_ismi = str(i) + ".jpg"
      if (os.path.isfile(klasor_adi + "/" + dosya_ismi)):
        dosyalar.append(dosya_ismi)
  else:
    print("klasör yok")
    exit()
  date =datetime.date.today() #bu log dosyası oluşturmak için kullandığım bir method önemli değil
  file = open("logfile_{}.txt".format(date), "w") #text dosyasına yazmak için filestream başlatıyorum
  file.write("{} KLASÖRÜNDEKİ DOSYALAR TEST EDİLİYOR!\n".format(klasor_adi)) #log dosyasına ilk yazdığı değer.
  print("{} KLASÖRÜNDEKİ DOSYALAR TEST EDİLİYOR!".format(klasor_adi))
  for dosya in dosyalar: #şimdi dosyaları tek tek inceliyorum
    dosya_adi = klasor_adi + "/" + dosya #full path alabilmek için
    image_data = cv2.imread(dosya_adi, cv2.IMREAD_GRAYSCALE) # dosyayı okuduk
    image_data = cv2.resize(image_data, (64, 64)) # resize ettik
    image_datas.append([dosya_adi, image_data]) #[dosyaismi,dosya_verisi] şeklinde topluyorum en son oluşan matrix şu şekilde olacak 
	#  [[dosya1,data1],
	#   [dosya2,data2],
	#   [dosya3,data3]]
	#burdan sonra da teker teker elimizdeki test datasını neural networke sokup tahminleri alıyoruz.
  for data in image_datas:
    dosya_ismi = data[0]
    dll = dosya_ismi
    img_data = data[1]
    img_data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([img_data])[0]
    if (model_out[np.argmax(model_out)] < 0.40): #buraya threshold koydum zira %0.02 ihtimalle bile bazen maksimum olarak döndüğünden ona notaymış gibi işlem yapıyor.
      label = "Model bulunamadı"
    else:
      label = labelling(np.argmax(model_out))
      dosyaisimleri_ve_tahminler.append([dosya_ismi, label])
    file.write("{} dosyası için tahminimiz : {} \n".format(dosya_ismi,label))
    print("{} dosyası için tahminimiz : {}".format(dosya_ismi,label))

  file.close() #streami kapatıyorum
  np.save("autocrop.npy",dosyaisimleri_ve_tahminler),
  print("AutoCrop.npy dosyası oluşturuldu, tahminler yazdırıldı.")
  
  
def testing_one_file():
  test_file = filedialog.askopenfilename()
  if(os.path.isfile(test_file)):
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
    if(model_out[np.argmax(model_out)] < 0.40):
      plt.title("Model bulunamadı")
    else:
      plt.title(labelling(np.argmax(model_out)))
    plt.imshow(orig)
    plt.show()

#argümanlı giriş için yaptığım basit bir sistem sorgusu.

if(sys.argv[1].split("=")[0] == "folder"):
  autocrop(sys.argv[1].split("=")[1])
else:
  testing_one_file()
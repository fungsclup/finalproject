import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm    
from dataset_creator import *
LR = 0.001 #Learning-rate
MODEL_NAME = './data/dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # modeli kaydetmeden önce ismini belirledim


#dosyalar yoksa bir dataset_creatorden fonksiyon çekiyorum varsa direk değişkene çekiyorum.
if os.path.exists(train_data_name) and os.path.exists(test_data_name):
	print("##################################### \n Veriler okunuyor... \n#####################################")
	train_data = np.load(train_data_name)
	test_data = np.load(test_data_name)
else:
	print("Veriler oluşturuluyor")
	train_data = dataset_train()
	test_data = dataset_test()

tf.reset_default_graph() # varsayılan grafiği temizliyorum.

#Network tanımlama
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
model = tflearn.DNN(convnet, tensorboard_dir='log')
##############################################

if os.path.exists('{}.meta'.format(MODEL_NAME)): #Model hali hazırda kayıtlıysa onun üzerinden devam ettiriyorum.
    model.load(MODEL_NAME)
    print('model loaded!')
############################# değişkenleri atadım, burda testide train datasetinden çekiyorum böylelikle öğrenirken classların ne olduğu belli oluyor
train = train_data  #supervised training (data,label)
test = train_data[250:1800]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1) # input için resimleri çekip yeniden şekillendirdim
Y = np.array([i[1] for i in train]) # bu da label tanımlaması

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1) #aynı şeyi test içinde yaptı
test_y = np.array([i[1] for i in test])

model.fit({'input': X}, {'targets': Y}, n_epoch=1000, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=100, show_metric=True, run_id=MODEL_NAME) # bu komutla yapılandırılmış modeldeki işlemi başlatıyoruz.
    
model.save(MODEL_NAME) # her bir adımda aynı modelin üzerine yazdırıyoruz.

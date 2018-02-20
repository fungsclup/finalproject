import numpy as np
import random
import os
import cv2
import tensorflow as tf
IMG_SIZE = 64
ostype = 1 #Windows = 1 Ubuntu = 2

# Hem masaüstü hemde laptopta çalıştırabilmek için yazdığım gereksiz bir koşul.
if ostype == 1:
	test_data_dir = "D:/PROJELER/examples/task3/test/"
	dataset_dir = 'D:/PROJELER/examples/task3/dataset/'
elif ostype == 2:
	test_data_dir = '/home/tarik/examples/task3/test/'
	dataset_dir = '/home/tarik/examples/task3/dataset/'

train_data_name = "./train_data.npy"
test_data_name = "./test_data.npy"
#  trainlenecek dataseti burda numpy arraye çeviriyorum

def dataset_train():
	train_data = []
	dirs = os.listdir(dataset_dir)
	for i in dirs:
		img_path = os.listdir(dataset_dir + '/' + i)
		if i == "dog": label = [1,0,0]
		elif i== "cat": label = [0,1,0]
		else : label = [0,0,1]
		for img in img_path:
			yol = dataset_dir + i +"/"+ img 
			image_data = cv2.imread(yol,cv2.IMREAD_GRAYSCALE)
			image_data = cv2.resize(image_data,(IMG_SIZE,IMG_SIZE))
			train_data.append([np.array(image_data),np.array(label)])
	random.shuffle(train_data)
	np.save(train_data_name,train_data)
	return train_data

#test datasetinide burda dönüştürüyorum.

def dataset_test():
  test_data = []
  for img in os.listdir(test_data_dir):
    path = os.path.join(test_data_dir, img)
    img_num = img.split('.')[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_data.append([np.array(img), img_num])
  random.shuffle(test_data)
  np.save(test_data_name, test_data)
  return test_data

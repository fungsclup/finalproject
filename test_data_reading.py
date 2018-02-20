import os 
import numpy as np 
import cv2
IMG_SIZE = 64
test_file = './test/1.jpg'
def testing_one_file(test_file):
	np_test_data = []
	test_file_data = cv2.imread(test_file,cv2.IMREAD_GRAYSCALE)
	test_file_data = cv2.resize(test_file_data,(IMG_SIZE,IMG_SIZE))
	np_test_data = [np.array(test_file_data),'5']
	return np_test_data
print(testing_one_file(test_file))

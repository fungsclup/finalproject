import numpy as np
import os
test_data_dir = '/home/tarik/examples/task3/test/'
file_list = os.listdir(test_data_dir)
num = 0

for ff in file_list:
	dosya_yolu = test_data_dir + ff
	print(dosya_yolu)
	yeni_isim = test_data_dir + str(num) + ".jpg"
	print(yeni_isim)
	os.rename(dosya_yolu,yeni_isim)
	num += 1


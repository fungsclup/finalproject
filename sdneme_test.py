from __future__ import division, print_function, absolute_import
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import numpy as np
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
test_data_image = testX[0].reshape(28,28)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_dir='log_deneme')
if(os.path.exists("./data/deneme.model.meta")):
  model.load("./data/deneme.model")
  print("model yükledi")
  print(np.shape(testX[0]))
  model_max = np.argmax(model.predict(testX[0].reshape(1,28,28,1)))
  print(model.predict(testX[0].reshape(1,28,28,1)))
  print(model_max)
  plt.imshow(test_data_image)
  plt.title = model_max
  plt.show()
else:
  model.fit({'input': X}, {'target': Y}, n_epoch=2,
            validation_set=({'input': testX}, {'target': testY}),
            snapshot_step=100, show_metric=True, run_id='deneme.model')
  model.save("./data/deneme.model")

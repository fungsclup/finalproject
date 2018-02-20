from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
from dataset_creator import dataset_train,dataset_test
model_name = "./github_kaydedilen.model"
train_data_name = "./train_data2.npy"
test_data_name = "./test_data2.npy"
IMG_SIZE = 64
MODEL_NAME = './data/github_ornegi_yeni_model.model'

tf.reset_default_graph()
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

if os.path.exists(train_data_name) and os.path.exists(test_data_name):
	print("##################################### \n Veriler okunuyor... \n#####################################")
	train_data = np.load(train_data_name)
	test_data = np.load(train_data_name)
else:
	print("Veriler oluşturuluyor")
	train_data = dataset_train()
	test_data = dataset_test()

train = train_data
test = train_data[250:1800]

train_X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
train_Y = np.array([i[1] for i in train])
test_X = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_Y = np.array([i[1] for i in test])

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 4096 # MNIST data input (img shape: 28*28)
num_classes = 3 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out
def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})
    return estim_specs
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_X}, y=train_Y,
    batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, steps=num_steps)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_X}, y=test_Y,
    batch_size=batch_size, shuffle=False)
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])



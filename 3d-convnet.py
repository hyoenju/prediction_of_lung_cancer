import tensorflow as tf
import numpy as np
import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
from sklearn.model_selection import train_test_split
import random
import pandas
import numpy as np
from sklearn import metrics, cross_validation

random.seed(42)

data_dir = './segmented_npy/'
patients = os.listdir(data_dir)

patients_data = None
for data in patients:
    patient = np.load(data_dir+data)
    if patients_data == None:
        patients_data = patient
    else:
        patients_data = np.vstack((patients_data, patient))

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

keep_rate = 0.6

x = tf.placeholder('float')
y = tf.placeholder('float')


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
#                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
#                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([2,2,2,1,32])),
           #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
           'W_conv2_1':tf.Variable(tf.random_normal([2,2,2,32,48])),
           'W_conv2':tf.Variable(tf.random_normal([2,2,2,48,64])),
           #                                  64 features
           'W_fc':tf.Variable(tf.random_normal([54080,1024])),
           'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
            'b_conv2_1':tf.Variable(tf.random_normal([48])),
           'b_conv2':tf.Variable(tf.random_normal([64])),
           'b_fc':tf.Variable(tf.random_normal([1024])),
           'out':tf.Variable(tf.random_normal([n_classes]))}

#                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2_1']) + biases['b_conv2_1'])
    conv2 = tf.nn.relu(conv3d(conv2, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

much_data = np.load('muchdata-50-50-20.npy')

x_data = patients_data[:, 0]
y_data = patients_data[:, 1]

# train_data, validation_data = train_test_split(patients_data, test_size=0.3)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
# train_data = much_data[:-50]
# validation_data = much_data[-50:]


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 100
    with tf.Session() as sess:
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        print("START")
        for epoch in range(hm_epochs):
            epoch_loss = 0

            # X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.2, random_state=42)

            for idx in range(len(X_train)):
                total_runs += 1
                try:
                    X = X_train[idx]
                    Y = y_train[idx]
                    # X = data[0]
                    # Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                # input tensor. Not sure why, will have to look into it. Guessing it's
                # one of the depths that doesn't come to 20.
                    pass
                #print(str(e))

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i for i in X_test], y:[i for i in y_test]}))

            # print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        # print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        print('Accuracy:',accuracy.eval({x:[i for i in X_test], y:[i for i in y_test]}))


        print('fitment percent:',successful_runs/total_runs)

        save_path = saver.save(sess, "./model_1.ckpt" )
        print("Model saved in file: %s" % save_path)
        #
# Run this locally:
train_neural_network(x)

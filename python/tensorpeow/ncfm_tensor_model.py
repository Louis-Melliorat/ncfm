# coding: utf-8

## Neural network structure : 4 convolutionnal layers + 1 fully connected relu + 1 fully connected softmax

# Example for (96x96) images

# · · · · · · · · · ·      (input data, 1-deep)                  X [batch, 96, 96, 1]
# @ @ @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x3=>4 stride 1     W1 [6, 6, 3, 4]          B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                        Y1 [batch, 96, 96, 4]
#   @ @ @ @ @ @ @ @ @ @    -- conv. layer 6x6x4=>6 stride 2      W2 [6, 6, 4, 6]          B2 [6]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                       Y2 [batch, 48, 48, 6]
#      @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W3 [5, 5, 6, 12]         B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶                                                       Y3 [batch, 24, 24, 12]
#       @ @ @ @ @ @       -- conv. layer 4*4*12=>24 stride 2     W4 [4,4,12,24]           B4 [24]
#                                                                Y3 [batch, 12, 12, 24] => reshaped to YY [batch, 12*12*24]
#
#       \x/x\x\x/        -- fully connected layer (relu)         W5 [6*6*24, 200]    B5 [200]   
#        · · · ·                                                 Y5 [batch, 200]
#        \x/x\x/         -- fully connected layer (softmax)      W6 [200, 8]         B6 [8]
#         · · ·                                                  Y [batch, 20]

import os
import glob
import cv2
import datetime
import time
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
import numpy as np 
import pandas as pd 
from subprocess import check_output
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss, classification_report, confusion_matrix
import tensorflow as tf
import math
import random


def weight_variable(shape):
    ini = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(ini)

def bias_variable(shape):
    ini = tf.constant(0.1, shape=shape)
    #ini = tf.ones(0.1, shape=shape)
    return tf.Variable(ini)

def conv2d(X, W, B, stride):
    x = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(x + B)

def conv2d_BN(X, W, B, tst, iter, stride):
    x = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME') + B
    Y_bn, update_ema = batchnorm(x, tst, iter, convolutional=True)
    return tf.nn.relu(Y_bn), update_ema

def batchnorm(Ylogits, is_test, iteration, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance)*100/101, lambda: variance)  # 100 = mini-batch size, to compute unbiased variance
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, 0.0, 1.0, bnepsilon)
    return Ybn, update_moving_everages

def run_training(img_size, X_train, Y_train, X_test, Y_test, test_data, iteration, batch_size
                    , dropout, augmentation, min_learning_rate, max_learning_rate, decay_speed):

    DISPLAY = True
    DISPLAY_ITER = 10 
    DISPLAY_STEP = iteration / DISPLAY_ITER
    TEST_SUMMARY = True
    TEST_EVAL = 50

    dim_x = img_size[0]
    dim_y = img_size[1]

    X = tf.placeholder(tf.float32, [None, dim_x, dim_y, 3])
    Y_ = tf.placeholder(tf.float32, [None, 8])
    pkeep = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    # test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)

    # 4 convolutional layers
    nb_sliding_window = 3
    dim_output_conv = dim_x / (2**nb_sliding_window)
    K = 4   # first 
    L = 6   # second 
    M = 12  # third 
    P = 24  # fourth
    N = 200 # fully connected layer

    W1 = weight_variable([6, 6, 3, K])
    B1 = bias_variable([K])

    W2 = weight_variable([6, 6, K, L])
    B2 = bias_variable([L])

    W3 = weight_variable([5, 5, L, M])
    B3 = bias_variable([M])

    W4 = weight_variable([4, 4, M, P])
    B4 = bias_variable([P])

    W5 = weight_variable([dim_output_conv * dim_output_conv * P, N])  #  W5 = weight_variable([6 * 6 * P, N]) for (48x48)
    B5 = bias_variable([N])

    W6 = weight_variable([N, 8])
    B6 = bias_variable([8])


    Y1bn, update_ema1 = conv2d_BN(X, W1, B1, tst, iter, stride=1)   # output is 48*48
    Y2bn, update_ema2 = conv2d_BN(Y1bn, W2, B2, tst, iter, stride=2)  # output is 24*24
    Y3bn, update_ema3 = conv2d_BN(Y2bn, W3, B3, tst, iter, stride=2)  # output is 12*12
    Y4bn, update_ema4 = conv2d_BN(Y3bn, W4, B4, tst, iter, stride=2)  # output is 6*6

    # reshape the output from the fourth convolution for the fully connected layer
    YY = tf.reshape(Y4bn, shape=[-1, dim_output_conv * dim_output_conv * P])  # YY = tf.reshape(Y4, shape=[-1, 6 * 6 * P])  for (48x48)
    Y5l =  tf.matmul(YY, W5) + B5
    Y5bn, update_ema5 = batchnorm(Y5l, tst, iter)
    Y5 = tf.nn.relu(Y5bn)
    Y5d = tf.nn.dropout(Y5, pkeep)
    Ylogits = tf.matmul(Y5d, W6) + B6  # fully connected softmax

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5)

    # The model
    with tf.name_scope("Model") as scope:
        Y = tf.nn.softmax(Ylogits)

    # Cross entropy
    with tf.name_scope("Cross_loss_entropy") as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
        cross_entropy = tf.reduce_mean(cross_entropy)*100
        tf.scalar_summary("Cross_loss_entropy", cross_entropy)

    # Optimizer / training step
    #learning_rate = 0.01
    with tf.name_scope("Train") as scope:
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        #train_step = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.898).minimize(cross_entropy)


    # Accuracy
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('Accuracy', accuracy)                  


    # Initializing the variables
    init = tf.initialize_all_variables()

    # Merge all summaries into a single operator
    merged_summary_op = tf.merge_all_summaries()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        for step in range(iteration):

            # --- Learning rate decreasing --- 0.002 / 0.0005 / 3500  
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-step/decay_speed)

            # --- Training step ---
            batch_X, batch_Y = next_batch(batch_size, X_train,Y_train)
            if augmentation:
                batch_X = data_augmentation_1(batch_X)
                batch_X = data_augmentation_2(batch_X)

            # the backpropagation training step
            #sess.run([train_step] , feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: dropout})
            sess.run([train_step] , feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: dropout})
            sess.run(update_ema, feed_dict={X: batch_X, Y_: batch_Y, tst: False, iter: step})


            if DISPLAY and (step % DISPLAY_STEP == 0 and step != 0):
                train_acc, train_entropy = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0})
                valid_acc, valid_entropy = sess.run([accuracy, cross_entropy], feed_dict={X: X_test, Y_: Y_test, tst: False, pkeep: 1.0})
                print('{0}| train log-loss {1:0.2f} | valid log-loss {2:.2f} | train acc {3:.2f} | valid acc {4:.2f}'.format(step,\
                    train_entropy/100, valid_entropy/100, train_acc, valid_acc))

        #print("\nTuning completed!")

        # --- Prediction ---
        cv_prediction = sess.run(Y, feed_dict={X: X_test, lr: learning_rate, tst: True, pkeep: 1.0})
        test_prediction = sess.run(Y, feed_dict={X: test_data, lr: learning_rate, tst: True, pkeep: 1.0})
        # --- Testing model ---
    return cv_prediction, test_prediction


def next_batch(batch_size,X,Y):
    
    idx = np.random.permutation(len(X))
    X_shuffle =[X[i] for i in idx]
    Y_shuffle =[Y[i] for i in idx]
    
    X_batch = np.array(X_shuffle[0:batch_size])
    Y_batch = np.array(Y_shuffle[0:batch_size]) 
    
    return X_batch, Y_batch
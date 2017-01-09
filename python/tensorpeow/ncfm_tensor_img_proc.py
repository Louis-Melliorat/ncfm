# coding: utf-8
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


def get_im_cv2(path,img_size):
    img = cv2.imread(path)
    resized = cv2.resize(img, img_size, interpolation = cv2.INTER_LINEAR)
    return resized


def load_train(img_size):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    #print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        #print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', '/notebooks/notebooks/NCFM/', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl,img_size)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
     
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test(img_size):
    path = os.path.join('..', '/notebooks/notebooks/NCFM/', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl,img_size)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

def read_and_normalize_train_data(img_size):
    train_data, train_target, train_id = load_train(img_size)

    #print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    #print('Reshape...')
    #train_data = train_data.transpose((0, 3, 1, 2))

    #print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target =  pd.get_dummies(train_target).as_matrix()

    print('Train shape:', train_data.shape)
    #print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id

def read_and_normalize_test_data(img_size):
    start_time = time.time()
    test_data, test_id = load_test(img_size)

    test_data = np.array(test_data, dtype=np.uint8)
    #test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    #print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def rotate(image, angle):
    image = tf.cast(image, tf.float32)
    angle = angle / 180 * math.pi
    shape = image.get_shape().as_list()
    assert len(shape) == 3, "Input needs to be 3D."
    image_center = np.array([x/2 for x in shape][:-1])

    coord1 = tf.cast(tf.range(shape[0]), tf.float32)
    coord2 = tf.cast(tf.range(shape[1]), tf.float32)

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [shape[1]])

    coord2_vec_unordered = tf.tile(coord2, [shape[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [shape[0], shape[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.pack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.expand_dims(tf.pack([tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), 0)
    rot_mat_inv = tf.cast(tf.reshape(rot_mat_inv, shape=[2, 2]), tf.float32)
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find neighbors in old image
    coord1_old_nn = coord_old_centered[0, :] + image_center[0]
    coord2_old_nn = coord_old_centered[1, :] + image_center[1]

    # Clip values to stay inside image coordinates
    outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, shape[0]-1), tf.less(coord1_old_nn, 0))
    outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, shape[1]-1), tf.less(coord2_old_nn, 0))
    outside_ind = tf.logical_or(outside_ind1, outside_ind2)

    coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
    coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.pack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    coord1_old_nn0 = tf.floor(coord1_old_nn)
    coord2_old_nn0 = tf.floor(coord2_old_nn)
    sx = coord1_old_nn - coord1_old_nn0
    sy = coord2_old_nn - coord2_old_nn0
    coord1_old_nn0 = tf.cast(coord1_old_nn0, tf.int32)
    coord2_old_nn0 = tf.cast(coord2_old_nn0, tf.int32)
    coord1_old_nn0 = tf.boolean_mask(coord1_old_nn0, tf.logical_not(outside_ind))
    coord2_old_nn0 = tf.boolean_mask(coord2_old_nn0, tf.logical_not(outside_ind))
    coord1_old_nn1 = coord1_old_nn0 + 1
    coord2_old_nn1 = coord2_old_nn0 + 1
    interp_coords = [
        ((1.-sx) * (1.-sy), coord1_old_nn0, coord2_old_nn0),
        (    sx  * (1.-sy), coord1_old_nn1, coord2_old_nn0),
        ((1.-sx) *     sy,  coord1_old_nn0, coord2_old_nn1),
        (    sx  *     sy,  coord1_old_nn1, coord2_old_nn1)
    ]

    interp_old = []
    for intensity, coord1, coord2 in interp_coords:
        intensity = tf.transpose(tf.reshape(intensity, [shape[1], shape[0]]))
        coord_old_clipped = tf.transpose(tf.pack([coord1, coord2]), [1, 0])
        interp_old.append((intensity, coord_old_clipped))

    channels = tf.split(2, shape[2], image)
    image_rotated_channel_list = list()
    for channel in channels:
        channel = tf.squeeze(channel)
        interp_intensities = []
        for intensity, coord_old_clipped in interp_old:
            image_chan_new_values = tf.gather_nd(channel, coord_old_clipped)

            channel_values = tf.sparse_to_dense(coord_new, [shape[0], shape[1]], image_chan_new_values,
                                                0, validate_indices=False)

            interp_intensities.append(channel_values * intensity)
        image_rotated_channel_list.append(tf.add_n(interp_intensities))

    image_rotated = tf.transpose(tf.pack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated

def shift_right(images, dev):
    return np.roll(images, dev, axis=1)

def shift_top(images, dev):
    return np.roll(images, dev, axis=0)

def preprocessingImage(img):
    img = tf.image.random_flip_left_right(img) # 2. Randomly flip the image horizontally.
    #img = rotate(img, 10)
    #img = tf.image.random_flip_up_down(img, seed=None) # Randomly flips an image vertically 
    #img = tf.image.resize_image_with_crop_or_pad(img, 24, 24) # 1. Crop the central [height, width] of the image.
    return img


def data_augmentation_1(batch):
    tf_img = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
    modimg = tf.map_fn(preprocessingImage,tf_img)
    batch = modimg.eval(feed_dict={tf_img: batch})
    return batch


def data_augmentation_2(batch):
    batch = shift_image(batch, 10, 0.5)
    return batch


def shift_image(batch,pixel,p):
    if random.random() > p:
        batch = shift_right(batch,pixel)
        batch = shift_top(batch,pixel)
    return batch


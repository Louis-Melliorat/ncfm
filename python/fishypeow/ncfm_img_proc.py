# Get modules
import os
import glob
import cv2
import datetime
import time
#import warnings4

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from matplotlib import pyplot as pl
from keras.utils import np_utils
from subprocess import check_output
print(check_output(["ls", "../../data"]).decode("utf8"))



def get_im_cv2(path, img_size=(48, 48)):
    img = cv2.imread(path)
    resized = cv2.resize(img, img_size)#, cv2.INTER_LINEAR)
    return resized


def load_train(img_size=(48, 48)):

    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()


    print('Read train images, size : {0}'.format(img_size))

    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', '..', 'data', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl, img_size)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test(img_size=(48, 48)):
    path = os.path.join('..', '..', 'data', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_size)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id



def read_and_normalize_train_data(img_size=(48, 48)):
    train_data, train_target, train_id = load_train(img_size)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id



def read_and_normalize_test_data(img_size=(48, 48)):
    start_time = time.time()
    test_data, test_id = load_test(img_size)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

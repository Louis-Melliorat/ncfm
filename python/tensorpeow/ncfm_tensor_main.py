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
from sklearn.model_selection import StratifiedKFold


# Import model
from ncfm_tensor_model import weight_variable, bias_variable, conv2d, conv2d_BN, batchnorm, run_training, next_batch
# Read load & transform data
from ncfm_tensor_img_proc import get_im_cv2, load_train, load_test, read_and_normalize_train_data, read_and_normalize_test_data
# data augmentation
from ncfm_tensor_img_proc import shift_right, shift_top, preprocessingImage, data_augmentation_1, data_augmentation_2, shift_image 
#ln /dev/null /dev/raw1394


def cross_validation(n_folds, random_state):
    num_fold = 0
    cv_score = 0
    model_list = []
    #skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in KFold(train_data.shape[0], n_folds=n_folds, shuffle=True, random_state=random_state):
    #for train_index, test_index in skf.split(train_data[:,0,0], train_target[:,0]):
        num_fold += 1
        print('\n > > > > > Start KFold number {} from {} < < < < < '.format(num_fold, n_folds))
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        print(X_train.shape)
        print(Y_train.shape)
        print(X_valid.shape)
        print(Y_valid.shape)
        # CV on valid data and save model prediction on test_data
        cv_prediction, test_prediction = run_training(img_size, X_train, Y_train, X_valid, Y_valid, test_data ,iteration, batch_size, 
                                                        dropout, augmentation,min_learning_rate,max_learning_rate,decay_speed)
        score = evaluate_model(np.array(Y_valid), cv_prediction)
        cv_score += score
        model_list.append(test_prediction)

    print('\n\n*** CV LOG LOSS {} ***'.format(cv_score / n_folds))
    # Average n_folds prediction
    average_pred = np.mean(model_list, axis=0)
    return average_pred

def evaluate_model(Y_true, Y_prediction):
    score = log_loss(Y_true, Y_prediction)
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    print('Classification report and confusion matrix')
    print(classification_report(Y_true.argmax(1), Y_prediction.argmax(1), target_names=folders))
    print(confusion_matrix(Y_true.argmax(1), Y_prediction.argmax(1)))
    print("*** Log loss test:" + str(score))
    return score

def create_submission(predictions, ID, name):
    sub = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    sub.loc[:, 'image'] = pd.Series(ID, index=sub.index)
    sub.to_csv(name, index=False)
    

if __name__ == '__main__':

    global_start = time.time()
    print('*******************************************************************')
    print('Starting job')
    print('*******************************************************************')
    print('Tensorflow version: {}'.format(tf.__version__))

    # Define parameters
    img_size=(96,96)
    iteration = 200
    batch_size = 100
    augmentation = False
    n_folds = 5
    random_state = 10   
    dropout = 0.40
    max_learning_rate = 0.002
    min_learning_rate = 0.0005
    decay_speed = 3500  
    subfile = 'submission.csv'

    # Load train and test data
    train_data, train_target, train_id = read_and_normalize_train_data(img_size)
    test_data, test_id = read_and_normalize_test_data(img_size)

    # Prediction & create submission
    average_pred = cross_validation(n_folds=n_folds, random_state=random_state)
    create_submission(average_pred, test_id, subfile)

    # Print model info
    print("--> ", subfile)
    print("\n*image size:"+str(img_size)+" steps:"+str(iteration)+" batch size:"+str(batch_size)+" n_folds:"+str(n_folds))
    print(" dropout:"+str(dropout)+" data augmentation:"+str(augmentation))
    print("max_lr:"+str(max_learning_rate)+" min_lr:"+str(min_learning_rate)+" decay:"+str(decay_speed))
    tot_time = str(datetime.timedelta(seconds=(round(time.time() - global_start,0)))).split('.')[0]
    print('*******************************************************************')
    print('Total execution time : {} '.format(tot_time))
    print('*******************************************************************')
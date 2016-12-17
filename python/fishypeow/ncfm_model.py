# Machine Larning Stuff
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.metrics import log_loss, classification_report
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator

def create_model(learning_rate=1e-2, dec=1e-6, moment=0.898, img_size=(48, 48)):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_size[0], img_size[1]), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(96, activation='relu',init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(16, activation='relu',init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=learning_rate, decay=dec, momentum=moment, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy')

    return model

def evaluate_model():
    '''
    TODO
    '''



def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import losses as losses 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
from keras.models import Model
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
np.random.seed(0)


def masked_loss_function(y_true, y_pred, mask_value = -1):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    return K.mean(K.square(y_true * mask - y_pred * mask))
    #return losses.mean_squared_error(y_true * mask, y_pred * mask)



def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df[['steering', 'qs']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """

    inp = Input(shape=(66, 200, 3))
    L1 = Lambda(lambda x: x/127.5-1.0)(inp)
    L2 = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(L1)
    L3 = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(L2)
    L4 = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(L3)
    L5 = Conv2D(64, (3, 3), activation='elu')(L4)
    L6 = Conv2D(64, (3, 3), activation='elu')(L5)
    L7 = Dropout(args.keep_prob)(L6)
    L8 = Flatten()(L7)
    L9 = Dense(100, activation='elu')(L8)
    L10 = Dense(75, activation='elu')(L9)
    
    A1 = Dense(50, activation='elu')(L10)
    A2 = Dense(20, activation='elu')(A1)
    
    V1 = Dense(50, activation='elu')(L10)
    V2 = Dense(1, activation='elu')(V1)
    
    Q = Lambda(lambda x: x[0][:] + x[1][:] - K.max(x[1][:]))([V2, A2])

    model = Model(inp, Q)

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')   
    
    model.compile(loss=masked_loss_function, optimizer=Adam(lr=args.learning_rate))

    history = model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.05)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=1)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=80)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()


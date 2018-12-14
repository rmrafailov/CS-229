import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras import backend as K
from keras.models import load_model, Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input

import utils
from utils import INPUT_SHAPE, batch_generator

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        mu = 0
        # sigma = 0.03491 #rads
        sigma = 0 #rads
        steering_angle = float(data["steering_angle"])+ np.random.normal(mu,sigma)
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            dis = np.array([-1.2106, -1.1053 , -1.0, -0.89473684, -0.78947368, -0.68421053, 
                            -0.57894737, -0.47368421, -0.36842105, -0.26315789, -0.15789474, -0.05263158,
                             0.05263158,  0.15789474,  0.26315789,  0.36842105,  0.47368421, 0.57894737,  
                             0.68421053,  0.78947368,  0.89473684,  1.0, 1.1053, 1.2106        ])
            idx = np.argmax(model.predict(image, batch_size=1))
            # print("model prediction")
            # print(model.predict(image, batch_size=1))
            # print(dis[idx])
            # input("...")
            event = np.random.choice([0,0/25,-0/25],1,p=[0.95,0.025,0.025])
            steering_angle = float(dis[idx]/1.2106+ event)
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {} {}'.format(steering_angle, throttle, speed, event))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # model = load_model(args.model)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(75, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(20))
    model.summary()

    # inp = Input(shape=(66, 200, 3))
    # L1 = Lambda(lambda x: x/127.5-1.0)(inp)
    # L2 = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(L1)
    # L3 = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(L2)
    # L4 = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(L3)
    # L5 = Conv2D(64, (3, 3), activation='elu')(L4)
    # L6 = Conv2D(64, (3, 3), activation='elu')(L5)
    # L7 = Dropout(0.5)(L6)
    # L8 = Flatten()(L7)
    # L9 = Dense(100, activation='elu')(L8)
    # L10 = Dense(75, activation='elu')(L9)
    
    # A1 = Dense(50, activation='elu')(L10)
    # A2 = Dense(20, activation='elu')(A1)
    
    # V1 = Dense(50, activation='elu')(L10)
    # V2 = Dense(1, activation='elu')(V1)
    
    # Q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]))([V2, A2])

    # model = Model(inp, Q)

    model.load_weights('dueling_40.h5')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

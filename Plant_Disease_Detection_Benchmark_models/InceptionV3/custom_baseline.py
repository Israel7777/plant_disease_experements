import os, glob

import tensorflow as tf
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

print('tf version: ', tf.__version__)

try:
    from .utils import get_nb_files, plot_training, train_model, TRAIN_DIR, NB_EPOCHS, BATCH_SIZE
except:
    from utils import get_nb_files, plot_training, train_model, TRAIN_DIR, NB_EPOCHS, BATCH_SIZE



# input image dimensions
IM_WIDTH, IM_HEIGHT = 100, 100
input_shape = (IM_WIDTH, IM_HEIGHT, 3)

def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1,1), name=None):
    """Convolution with batch normalization and relu activation"""

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,
               name=conv_name, use_bias=False)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)

    return x

def Inceptionv3(nb_classes, input_tensor=None, input_shape=None):
    """InceptionV3 model"""

    if input_tensor is None:
        input = Input(shape=input_shape)
    else:
        input = input_tensor

    # starting stem of inceptionv3 architecture
    print('conv: ', input.shape)
    x = conv2d_bn(input, 32, 3, padding='valid', strides=(2,2))
    print('conv: ', x.shape)
    x = conv2d_bn(x, 32, 3, padding='valid')
    print('conv: ', x.shape)
    x = conv2d_bn(x, 64, 3)
    print('pool: ', x.shape)
    x = MaxPooling2D(3, strides=(2,2))(x)
    print('conv: ', x.shape)
    x = conv2d_bn(x, 80, 3, padding='valid')
    print('conv: ', x.shape)
    x = conv2d_bn(x, 192, 3, padding='valid', strides=(2,2))
    print('conv: ', x.shape)
    x = conv2d_bn(x, 288, 3)

    # 3 inceptions with kernel 5*5 factorized to two deep 3*3
    for i in range(3):
        conv1x1 = conv2d_bn(x, 192, 1)

        conv3x3 = conv2d_bn(x, 128, 1)
        conv3x3 = conv2d_bn(conv3x3, 192, 3)

        conv3x3dbl = conv2d_bn(x, 128, 1)
        conv3x3dbl = conv2d_bn(conv3x3dbl, 128, 3)
        conv3x3dbl = conv2d_bn(conv3x3dbl, 192, 3)

        pool = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
        pool = conv2d_bn(pool, 192, 1)

        x = concatenate([conv1x1, conv3x3, conv3x3dbl, pool], name='inception1_mix'+str(i))

    # 5 inceptions with kernels factorized into deep 1*n and n*1 rather than n*n
    for i in range(5):
        conv1x1 = conv2d_bn(x, 384, 1)

        conv3x3 = conv2d_bn(x, 320, 1)
        conv3x3 = conv2d_bn(conv3x3, 384, (1,3))
        conv3x3 = conv2d_bn(conv3x3, 384, (3,1))

        conv3x3dbl = conv2d_bn(x, 320, 1)
        conv3x3dbl = conv2d_bn(conv3x3dbl, 320, (1,3))
        conv3x3dbl = conv2d_bn(conv3x3dbl, 320, (3,1))
        conv3x3dbl = conv2d_bn(conv3x3dbl, 384, (1,3))
        conv3x3dbl = conv2d_bn(conv3x3dbl, 384, (3,1))

        pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = conv2d_bn(pool, 384, 1)

        x = concatenate([conv1x1, conv3x3, conv3x3dbl, pool], name='inception2_mix' + str(i))

    # 2 inceptions with kernels factorized into wide 1*n and n*1 rather than n*n
    for i in range(2):
        conv1x1 = conv2d_bn(x, 512, 1)

        conv3x3 = conv2d_bn(x, 440, 1)
        conv1x3 = conv2d_bn(conv3x3, 512, (1, 3))
        conv3x1 = conv2d_bn(conv3x3, 512, (3, 1))

        conv3x3dbl = conv2d_bn(x, 440, 1)
        conv3x3dbl = conv2d_bn(conv3x3dbl, 440, (3, 3))
        conv1x3dbl = conv2d_bn(conv3x3dbl, 512, (1, 3))
        conv3x1dbl = conv2d_bn(conv3x3dbl, 512, (3, 1))

        pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = conv2d_bn(pool, 512, 1)

        x = concatenate([conv1x1, conv1x3, conv3x1, conv1x3dbl, conv3x1dbl, pool], name='inception3_mix' + str(i))

    x = MaxPooling2D((8,8), padding='same')(x)

    flattened = Flatten()(x)
    print("flat shape: ", flattened.shape)
    outputs = Dense(nb_classes, activation='softmax')(flattened)
    print("output shape: ", outputs.shape)

    model = Model(inputs=input, outputs=outputs)

    return model


model = Inceptionv3(38, input_shape=input_shape)
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_model(model)
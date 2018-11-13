from __future__ import absolute_import

import argparse
import os

import numpy as np
import glob

import sys
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('tf version: ', tf.__version__)

try:
    from .utils import get_nb_files, plot_training, TRAIN_DIR, VAL_DIR, NB_EPOCHS, BATCH_SIZE
except SystemError:
    from utils import get_nb_files, plot_training, TRAIN_DIR, VAL_DIR, NB_EPOCHS, BATCH_SIZE

MODEL_STORE_TEMPLATE = "../Models/InceptionV3-finetuning-{}.h5"
MODEL_LOG_TEMPLATE = "inception_scratch_log_{}.csv"

batch_size = 128
epochs = 30
num_classes = len(glob.glob(TRAIN_DIR + "/*"))
nb_train_samples = get_nb_files(TRAIN_DIR)
nb_val_samples = get_nb_files(VAL_DIR)

# input image dimensions
IM_WIDTH, IM_HEIGHT = 200, 200
input_shape = (IM_WIDTH, IM_HEIGHT, 3)
tf.keras.Sequential

def train(args):
    identifier = input('Enter identifying model name: ')
    CSV_LOG_FILE = MODEL_LOG_TEMPLATE.format(identifier)

    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    if args.output_model_file is not None:
        output_model_file = args.output_model_file
    else:
        output_model_file = MODEL_STORE_TEMPLATE.format(identifier)


    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size)
    validation_generator = test_datagen.flow_from_directory(VAL_DIR, target_size=(IM_WIDTH, IM_HEIGHT),
                                                      batch_size=batch_size)

    iv3 = InceptionV3(input_shape=(IM_HEIGHT, IM_WIDTH, 3), weights=None,
                      include_top=True, classes=nb_classes)
    iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        EarlyStopping(min_delta=0.001, patience=10),
        CSVLogger(CSV_LOG_FILE)
    ]
    history_ft = iv3.fit_generator(train_generator, epochs=nb_epoch, steps_per_epoch=nb_train_samples // batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=nb_val_samples // batch_size,
                                     callbacks=callbacks)

    iv3.save(output_model_file)

    plot_training(history_ft)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default=TRAIN_DIR)
    a.add_argument("--val_dir", default=VAL_DIR)
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BATCH_SIZE)
    a.add_argument("--output_model_file")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)

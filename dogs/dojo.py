# dogs.dojo.model.py

import os
import sys
from subprocess import run

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # !{ERROR,WARNING,INFO}
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from tensorflow import keras
from tensorflow import config as tf_config
from tensorflow.data.experimental import AUTOTUNE # AAAAAAAAKON

from efficientnet.tfkeras import EfficientNetB0, preprocess_input

from db import api as db_api

physical_devices = tf_config.list_physical_devices('GPU')
tf_config.experimental.set_memory_growth(physical_devices[0], True)

keras.mixed_precision.set_global_policy('mixed_float16')

CLASS_NAMES = ['laying', 'sitting', 'standing']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32

TRAINING_DIR = os.getenv('TRAINING_DIR')
VALIDATION_DIR = os.getenv('VALIDATION_DIR')
CHECKPOINTS_DIR = os.getenv('CHECKPOINTS_DIR')
LOG_DIR = os.getenv('LOG_DIR')
SAVE_DIR = os.getenv('SAVED_MODELS')

IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)
SEED = 1337

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SaveBestWeights(keras.callbacks.ModelCheckpoint):
    def __init__(self, model_name):
        self.filepath = CHECKPOINTS_DIR + model_name + '-best_weights.hdf5'

        super(SaveBestWeights, self).__init__(
            filepath=self.filepath,
            verbose=1, save_weights_only=True, save_best_only=True,
            monitor='val_loss', mode='min', save_freq='epoch')

def start(model_name, hyperparameters):
    #train_ds, val_ds = get_training_sets()
    train_ds, val_ds, cw = db_api.get_dojo_datasets()
    train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    class_weights = None

    # COARSE TRAINING
    coarse_model = build_model_for_coarse_training(model_name, hyperparameters)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=hyperparameters.coarse_decay_factor,
        patience=2, cooldown=2,
        min_lr=hyperparameters.coarse_learning_rate_min)
    stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='auto', patience=10, verbose=0)
    logfile = LOG_DIR + model_name + '.log'
    csv_logger = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    callbacks=[SaveBestWeights(model_name), csv_logger, reduce_lr, stop_callback]
    train_model(train_ds, val_ds, class_weights, coarse_model, model_name, 
                epochs=hyperparameters.coarse_epochs,
                callbacks=callbacks)
    # FINE TUNING
    fine_model_name = model_name + '-fine_tuned'
    fine_model = build_model_for_fine_tuning(fine_model_name, hyperparameters)
    fine_model.load_weights(CHECKPOINTS_DIR + model_name + '-best_weights.hdf5')
    logfile = LOG_DIR + fine_model_name + '.log'
    csv_logger = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    callbacks=[SaveBestWeights(fine_model_name), csv_logger]
    train_model(train_ds, val_ds, class_weights, fine_model, fine_model_name,
                epochs=hyperparameters.fine_epochs,
                callbacks=callbacks)

def build_model_for_fine_tuning(model_name, hyperparameters):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hyperparameters.fine_learning_rate,
        decay_steps=hyperparameters.fine_decay_steps,
        decay_rate=hyperparameters.fine_decay_rate)
    return build_model(model_name, 
                       lr_schedule,
                       hyperparameters.fine_dropout,
                       hyperparameters.fine_l2_regularization,
                       base_model_trainable=True)

def build_model_for_coarse_training(model_name, hyperparameters):
    return build_model(model_name,
                       hyperparameters.coarse_learning_rate_initial,
                       hyperparameters.coarse_dropout,
                       hyperparameters.coarse_l2_regularization)
    
def build_model(model_name, learning_rate, dropout,
                l2_regularization, base_model_trainable=False):
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = preprocess_input(inputs)
    base_model = EfficientNetB0(
        input_tensor=x, weights='imagenet',
        include_top=False, drop_connect_rate=dropout)

    base_model.trainable = base_model_trainable
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(
        NUM_CLASSES,
        kernel_regularizer=keras.regularizers.L2(l2_regularization), 
        activation='softmax',
        dtype='float32')(x)

    model = keras.Model(inputs, outputs, name=model_name)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def train_model(train_ds, val_ds, class_weights, model, model_name, epochs, callbacks):
    # model.summary()
    history = model.fit(
            train_ds,
            callbacks=callbacks,
            class_weight=class_weights,
            epochs=epochs, 
            use_multiprocessing=True,
            validation_data=val_ds, 
            verbose=2, 
            workers=6)
    model.save(SAVE_DIR + model_name)
    return min(history.history['val_loss'])


def get_training_sets():
    train_ds = keras.preprocessing.image_dataset_from_directory(
        directory=TRAINING_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)
    
    val_ds = keras.preprocessing.image_dataset_from_directory(
        directory=VALIDATION_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)
    return train_ds, val_ds

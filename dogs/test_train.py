import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # !{ERROR,WARNING,INFO}
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
import tensorflow.keras.preprocessing.image as Kimage
keras.mixed_precision.set_global_policy('mixed_float16')

from tensorflow.data.experimental import AUTOTUNE # AAAAAAAAKON
from efficientnet.tfkeras import EfficientNetB0, preprocess_input
from imgaug import augmenters as iaa

import db.api as db_api
from db.hdf5_stuff import dojo_store, dojo_read

CLASS_NAMES = ['laying', 'sitting', 'standing']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)

HYPERPARAMETERS = {
        'coarse_epochs': 1,
        'fine_learning_rate': 1e-4,
        'fine_decay_rate': 0.5,
        'fine_decay_steps': 10000,
        'fine_dropout': 0.9,
        'fine_epochs': 10,
}

def build_model(model_name, learning_rate=1e-3, l2_rate=1e-2, dropout=0.5, base_model_trainable=False):
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = preprocess_input(inputs)
    base_model = EfficientNetB0(input_tensor=x, weights='imagenet', include_top=False)

    base_model.trainable = base_model_trainable
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(
        NUM_CLASSES,
        kernel_regularizer=keras.regularizers.L2(l2_rate), 
        activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs, name=model_name)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def rand_aug(image):
    #seq = iaa.Sequential([
    #    iaa.Flipud(0.25),
    #    iaa.RandAugment(n=2, m=9),
    #])
    return tf.cast(iaa.RandAugment(n=2, m=9).augment_image(image=image), tf.float32)


def get_training_sets(train_limit=-1, val_limit=-1):
    #train_ds = tf.data.Dataset.from_generator(
    #    lambda: (i for i in image_gen('train', train_limit, aug=True)),
    #    output_signature=(
    #        tf.TensorSpec(shape=(BATCH_SIZE, *IMG_SIZE, 3), dtype=tf.float32),
    #        tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32)
    #    )
    #)
    #val_ds = tf.data.Dataset.from_generator(
    #    lambda: (i for i in image_gen('dev', val_limit, aug=False)),
    #    output_signature=(
    #        tf.TensorSpec(shape=(BATCH_SIZE, *IMG_SIZE, 3), dtype=tf.float32),
    #        tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32)
    #    )
    #)
    

    images = dojo_read(train_limit, val_limit)
    x_train, y_train = images[0]
    x_test, y_test = images[1]

    steps = len(x_train) // BATCH_SIZE

    @tf.function(input_signature=[tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.uint8)])
    def do_rand_aug(images):
        aug = tf.numpy_function(rand_aug, [images], tf.float32)
        return aug

    #x_train = do_rand_aug(tf.cast(x_train, tf.uint8))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_ds = train_ds.shuffle(len(x_train))
    train_ds = train_ds.repeat(11)

    # RandAugment
    train_ds = train_ds.map(lambda x, y: (do_rand_aug(tf.cast(x, tf.uint8)), y), num_parallel_calls=AUTOTUNE)

    # Tensorflow augmenters
    #train_ds = train_ds.map(lambda x, y: (tf.image.random_brightness(x, 0.2), y), num_parallel_calls=AUTOTUNE)
    #train_ds = train_ds.map(lambda x, y: (tf.image.random_contrast(x, 0.2, 0.5), y), num_parallel_calls=AUTOTUNE)
    #train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    #train_ds = train_ds.map(lambda x, y: (tf.image.random_flip_up_down(x), y), num_parallel_calls=AUTOTUNE)
    #train_ds = train_ds.map(lambda x, y: (tf.image.random_hue(x, 0.2), y), num_parallel_calls=AUTOTUNE)
    
    val_ds = train_ds.shuffle(len(x_test))
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    return train_ds, val_ds, steps

def train():
    train_ds, val_ds, steps = get_training_sets()
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    coarse = build_model('coarse-test')
    coarse.fit(train_ds, epochs=HYPERPARAMETERS['coarse_epochs'],
               validation_data=val_ds, shuffle=False, steps_per_epoch=steps,
               use_multiprocessing=True, workers=6, verbose=1)

    weights = coarse.layers[-1].get_weights()
    fine = build_model('fine-test',
                       learning_rate=HYPERPARAMETERS['fine_learning_rate'],
                       l2_rate=1e-1,
                       dropout=HYPERPARAMETERS['fine_dropout'],
                       base_model_trainable=True)
    fine.layers[-1].set_weights(weights)
    fine.fit(train_ds, epochs=HYPERPARAMETERS['fine_epochs'],
             validation_data=val_ds, shuffle=False, steps_per_epoch=steps,
             use_multiprocessing=True, workers=6, verbose=1)
    
    print(max(fine.history.history['val_categorical_accuracy']))

if __name__ == '__main__':
    train()

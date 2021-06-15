import h5py
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as Kimage

from .api import filepaths_by_label

# TODO remove image, label at index (remove incorrect labels)

def resize_image(image, target_size, interpolation='lanczos5'):
    img = Kimage.img_to_array(Kimage.load_img(image))
    img = Kimage.smart_resize(img, size=target_size, interpolation=interpolation)
    return img


def dojo_store(image_size=(224,224)):
    # Store resized dojo images in high perf cache file
    hdf5_dir = Path('/data/hdf5')
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    label_case = {
        1: 0,
        2: 1,
        3: 2
    }

    file_ = h5py.File(hdf5_dir / "dojo.h5", "w")
    for set_ in ['train', 'dev']:
        image_l = []
        label_l = []
        for class_ in ['laying', 'sitting', 'standing']:
            images_and_labels = filepaths_by_label(class_, set_)
            
            for image in images_and_labels:
               image_l.append(resize_image(image[0], target_size=image_size))
               label_l.append(label_case.get(image[1]))
        
        images = tf.stack(image_l)
        labels = tf.stack(label_l)

        #labels = tf.keras.utils.to_categorical(labels, num_classes=3)

        file_.create_dataset(
            f"{set_}_images", tf.shape(images), h5py.h5t.IEEE_F32LE, data=images)
        file_.create_dataset(
            f"{set_}_labels", tf.shape(labels), h5py.h5t.IEEE_F32LE, data=labels)

    file_.close()


def dojo_read(train_limit=-1, dev_limit=-1, seed=32):
    hdf5_dir = Path('/data/hdf5')
    file_ = h5py.File(hdf5_dir / "dojo.h5", "r+")
   
    train_images = file_['train_images']
    train_labels = file_['train_labels']
    dev_images = file_['dev_images']
    dev_labels = file_['dev_labels']

    print('1')
   
    if train_limit != -1:
        random.seed(seed)
        rand = random.sample(range(0, len(train_images)), train_limit) 
        train_images = tf.stack([train_images[n] for n in rand])
        train_labels = tf.stack([train_labels[n] for n in rand])
        print('2')
    if dev_limit != -1:
        random.seed(seed)
        rand = random.sample(range(0, len(dev_images)), dev_limit) 
        dev_images = tf.stack([dev_images[n] for n in rand])
        dev_labels = tf.stack([dev_labels[n] for n in rand])
        print('3')

    
    print(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape)
    return (train_images, train_labels), (dev_images, dev_labels)

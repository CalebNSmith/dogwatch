# dogs/tf_dogs.py

import os
import sys
from subprocess import run
import numpy as np
from PIL import Image
import threading
import cv2


import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # !{ERROR,WARNING,INFO}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # !{ERROR,WARNING,INFO}
import tensorflow
tensorflow.get_logger().setLevel('INFO')
from tensorflow import keras, expand_dims
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.config import list_physical_devices
from tensorflow.config.experimental import set_memory_growth
from tensorflow.keras.applications.efficientnet import preprocess_input

# physical_devices = list_physical_devices('GPU')
# set_memory_growth(physical_devices[0], True)

UNLABELED = os.getenv('IMG_DIR') + '/unlabeled/'
IMG_SIZE = (224, 224)
CLASS_NAMES=['laying', 'sitting', 'standing']
CLASS_CUTOFFS={0: 0.9, 1: 0.99, 2: 0.95}
NUM_THREADS = 6

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_and_resize(chunked_list, results, thread_num):
    images = []
    img_filenames = []
    for fname in chunked_list:
        img = os.path.join(UNLABELED, fname)
        try:
            img = image.load_img(img, target_size=(224,224))
        except Exception as e:
            print(e)
            continue
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
        img_filenames.append(fname)

    results[thread_num] = {'images': images, 'img_filenames': img_filenames}
    return True

model = load_model('../../saved_models/reduce_on_plateau-fine_tuned')
model.summary()

results = {}
save_threads = []
new_img_filenames = list(chunks(img_filenames, max(len(img_filenames) // NUM_THREADS, 1)))
for n, sublist in enumerate(new_img_filenames):
    t = threading.Thread(name=str(n), target=load_and_resize, args=(sublist, results, n))
    save_threads.append(t)
    t.start()

for t in save_threads:
    t.join()

img_filenames = []
images = []

for n in results:
    images.extend(results[n]['images'])
    img_filenames.extend(results[n]['img_filenames'])

# stack up images list to pass for prediction
images = np.vstack(images)
images = preprocess_input(images)
classes = model.predict(images, batch_size=1024, verbose=1)

#print(classes)
x = list(classes)
z = list(zip(img_filenames, x))
num_wrong = 0
num_predicted = 0

with open(UNLABELED + 'predicks.txt', 'w') as f:
    for p in z:
        predick = p[1]
        if (np.max(predick) < CLASS_CUTOFFS[np.argmax(predick)]):
            continue

        num_predicted += 1
        if (np.argmax(predick) != 1):
            num_wrong += 1
        #f.write('%s %s %s\n' % (p[0], CLASS_NAMES[np.argmax(predick)], str(list(predick))))

print("Num total: %d" % (len(z)))
print("Num predicted: %d" % (num_predicted))
print("Predicted ratio: %.2f" % (float(num_predicted) / len(z)))
print("Num right: %d" % (num_predicted - num_wrong))
print("Num wrong: %d" % (num_wrong))
print("Wrong ratio: %.2f" % (float(num_wrong) / num_predicted))

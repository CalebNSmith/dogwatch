# human_touch.py
# Usage: python human_touch.py <owner> <number> <label>
# Example: python human_touch.py dan 1000 sitting
# ^ will run until 1000 predictions

import glob
import numpy as np
import os
import sys
import shutil
import threading
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # !{ERROR,WARNING,INFO}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # !{ERROR,WARNING,INFO}

import tensorflow as tf
from tensorflow import keras, config

#physical_devices = config.list_physical_devices('GPU')
#config.experimental.set_memory_growth(physical_devices[0], True)

from db import DojoImage
from db.dbwrapper import Database
from analysis import heatmap

IMAGE_SIZE = 224
INT_LABELS = {'laying': 0, 'sitting': 1, 'standing': 2}
MODEL_NAME = 'dropout_30_dropout_90-fine_tuned' # TODO get_current_best_model()
LAST_CONV_LAYER_NAME = 'top_activation'
CLASSIFIER_LAYER_NAMES = [
    'global_average_pooling2d_1',
    'batch_normalization_1',
    'dense_1',
]
CUTOFF = 0.85

def batch_resize(batch, n, resized):
    # batch is list of DojoImage objects
    for image in batch:
        try:
            image.resize(target_size=IMAGE_SIZE)
            image.as_array = np.array([image.as_array])
            resized.append(image)
        except Exception as e:
            print(e)

    print('BATCH %d RESIZED' % (n,))

def chunks(lst, n):
    stride = math.ceil(len(lst) / n)
    for offset in range(0, len(lst), stride):
        yield lst[offset:offset+stride]

def prep_filesystem(out_dir, out_db, out_images, out_heatmaps):
    # mkdir img/unlabeled/$owner if not exists or remove all files and leave skeleton
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_images)
        os.mkdir(out_heatmaps)
    else:
        for f in glob.glob(out_images + '/*'):
            os.remove(f)
        for f in glob.glob(out_heatmaps + '/*'):
            os.remove(f)

        try:
            os.remove(out_db)
        except Exception as e:
            print(e)

        try:
            os.remove(out_dir + '/labeler.py')
        except Exception as e:
            print(e)

def get_for_checkout(model, owner, number, str_label):
    for_check_out = []
    for_check_out_arrays = []

    while len(for_check_out) < number:
        batch_size = min(2048, 6 * (number - len(for_check_out)))
        print('Loading %d unlabeled images...' % batch_size, end=' ')
        unlabeled_images = DojoImage().unlabeled_images(number=batch_size)
        print('Done.')

        resized = []
        threads = []
        for n, batch in enumerate(chunks(unlabeled_images, 5)):
            resized.append([])
            t = threading.Thread(name=n, target=batch_resize, args=(batch, n, resized[n]))
            threads.append(t)
            t.start()
        
        print('Resizing images...')
        for t in threads:
            t.join()

        # load $current_best_model and make predictions
        print('Making class predictions...')
        resized = [item for sublist in resized for item in sublist]
        vstacked = np.vstack([image.as_array for image in resized])
        preds = model.predict(vstacked, batch_size=min(512, batch_size), verbose=1)

        for n, p in enumerate(preds):
            if len(for_check_out) >= number:
                break

            if np.argmax(p) == INT_LABELS[str_label] and np.max(p) > CUTOFF:
                DojoImage().check_out_image(resized[n], owner)
                for_check_out.append((resized[n], p))
                for_check_out_arrays.append(vstacked[n])
            else:
                # temporalily check out images so not just looping through same images over and over
                DojoImage().check_out_image(resized[n], 'tmp')

        print('Found %d/%d %s images\n' % (len(for_check_out), number, str_label))


    print('Making heatmaps...', end=' ')
    heatmaps = heatmap.make_heatmap(model, LAST_CONV_LAYER_NAME,
                                    CLASSIFIER_LAYER_NAMES,
                                    tf.convert_to_tensor(for_check_out_arrays))
    heatmaps = [keras.preprocessing.image.array_to_img(hm) for hm in heatmaps]
    for_check_out = zip(for_check_out, heatmaps)
    print('Done.')

    return for_check_out

def save_images_and_predictions(out_db, str_label, for_check_out):
    # save predictions in img/unlabeled/$owner/predictions.db
    with Database(out_db) as db:
        db.query(
        """ CREATE TABLE prediction (
                image_id INTEGER PRIMARY KEY,
                laying REAL NOT NULL,
                sitting REAL NOT NULL,
                standing REAL NOT NULL,
                human_label TEXT NULL DEFAULT NULL,
                model_label TEXT NOT NULL,
                human_dataset TEXT NOT NULL DEFAULT 'training'
            )""")

        for (image, preds), hm in for_check_out:
            db.insert(
            """ INSERT INTO prediction 
                (image_id, laying, sitting, standing, model_label)
                VALUES (?, ?, ?, ?, ?)""",
                (image.image_id, preds[0].astype(float), preds[1].astype(float), preds[2].astype(float), str_label))
            
            # save $resized_images in img/unlabeled/$owner/images
            filepath=out_dir + '/images/' + str(image.image_id) + '.jpeg'
            heat_filepath=out_dir + '/heatmaps/' + str(image.image_id) + '.jpeg'
            image.save(new_filepath=filepath)
            hm.save(heat_filepath)

def main(owner, out_dir, out_db, out_images, out_heatmaps, number, str_label):
    print('Preparing filesystem...', end=' ')
    prep_filesystem(out_dir, out_db, out_images, out_heatmaps)
    print('Done.')

    print('Loading model...', end=' ')
    model = keras.models.load_model('../saved_models/' + MODEL_NAME)
    model.compile()
    print('Done.')

    for_check_out = get_for_checkout(model, owner, number, str_label)
    save_images_and_predictions(out_db, str_label, for_check_out)

    # free up temporalily checked out images 
    DojoImage().un_check_out_images('tmp')

    # copy labeler.py to out_dir and print scp command to pull images in
    shutil.copy('labeler.py', out_dir)
    print()
    print('rm -rf ~/unlabeled ; mkdir -p ~/unlabeled ; scp -r ubuntu-ml:/home/dan/dogs/img/unlabeled/%s ~/unlabeled' % (owner,))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print()
        print('MOAR ARGS!!')
        print('Usage: python human_touch.py <owner> <number> <label>')
        print('Example: python human_touch.py dan 1000 sitting')
        print()
        exit()

    owner = sys.argv[1]
    out_dir = os.getenv('IMG_DIR') + '/unlabeled/' + owner
    out_db = out_dir + '/predictions.db'
    out_images = out_dir + '/images'
    out_heatmaps = out_dir + '/heatmaps'
    number = int(sys.argv[2])
    str_label = sys.argv[3]
    main(owner, out_dir, out_db, out_images, out_heatmaps, number, str_label)

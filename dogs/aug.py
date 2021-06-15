import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # !{ERROR,WARNING,INFO}
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # !{ERROR,WARNING,INFO}


import numpy as np
import threading
from PIL import Image

from imgaug import augmenters as iaa

from db.dbwrapper import Database
from db.dojo_image import DojoImage

#physical_devices = config.list_physical_devices('GPU')
#config.experimental.set_memory_growth(physical_devices[0], True)

IMG_DIR = os.getenv('IMG_DIR')
NUM_AUGS = 2
LABELS = ['laying', 'sitting', 'standing']
TRAINING_DIR = os.getenv('TRAINING_DIR')
VALIDATION_DIR = os.getenv('VALIDATION_DIR')

def aug_and_save(label, seq):
    print('Aug %s' % (label,))
    work_set = training_sets['training'][label]
    resized = []
    for image in work_set:
        image.resize(target_size=224)
        image.as_array = np.array([image.as_array])
        resized.append(image)

    vstacked = np.vstack([image.as_array.astype(np.uint8) for image in resized])
    for aug_pass in range(NUM_AUGS):
        images_aug = seq(images=vstacked)
        for n, aug in enumerate(images_aug):
            image = resized[n]
            image.as_array = aug
            #filepath = IMG_DIR + '/testing/%s/%s/%s_rand%d.jpeg' % (save_dir, label, image.image_id, aug_pass)
            filepath = '%s/%s/%s_rand%d.jpeg' % (TRAINING_DIR, label, image.image_id, aug_pass)
            image.save(filepath)

def resized(images):
    def calc_box(width, height):
        crop_height = (width * TARGET_SIZE) / TARGET_SIZE
        crop_width = (height * TARGET_SIZE) / TARGET_SIZE
        crop_height = min(height, crop_height)
        crop_width = min(width, crop_width)
        crop_box_hstart = (height - crop_height) / 2
        crop_box_wstart = (width - crop_width) / 2
        return (crop_box_wstart, crop_box_hstart, crop_width, crop_height)

    for i in images:
        w, h = i.width, i.height
        size = (TARGET_SIZE, TARGET_SIZE)
        resized = i.resize(size, box=calc_box(w, h), resample=Image.LANCZOS)
        yield (resized, i.filename)

def just_resize(label):
    print('Resize %s' % (label,))
    work_set = training_sets['dev'][label] # Change as get more validation images
    for image in work_set:
        image.resize(target_size=224)
        filepath = '%s/%s/%s.jpeg' % (VALIDATION_DIR, label, image.image_id)
        image.save(filepath)

if __name__ == '__main__':
    seq = iaa.Sequential([
        iaa.Flipud(0.25),
        iaa.RandAugment(n=2, m=9),
    ])
    
    training_sets = DojoImage().get_training_sets() # can pass number=N to set a limit
    threads = []
    for label in LABELS:
        t = threading.Thread(name=label + '_t', target=aug_and_save, args=(label, seq))
        threads.append(t)
        t.start()
        vt = threading.Thread(name=label + '_v', target=just_resize, args=(label,))
        threads.append(vt)
        vt.start()

    for t in threads:
        t.join()

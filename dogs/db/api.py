from imgaug import augmenters as iaa
import numpy as np
from peewee import *
from pathlib import Path
from PIL import Image as PilImage
import random

from tensorflow import keras
import tensorflow.keras.preprocessing.image as Kimage

from .base_models import (
    db,
    Epoch,
    Image,
    ImageContext,
    Label,
    Prediction,
    Project,
    TrainingRun,
)
from .dog_models import (
    Breed,
    BreedIndex,
    Dog,
    DogImage,
)


def query_from_file(sqlfile):
    with open(sqlfile, 'r') as f:
        sql = f.read()
    return sql


def image_path(filename):
    base_path = Path('/data/images')
    return base_path.joinpath(filename[0], filename[1], filename)


def filepaths_by_label(label, training_set, limit=-1):
    sql = query_from_file('db/sql/filepaths_by_label.sql')
    training_set = set_enum(training_set)
    query = db.execute_sql(sql, (label, training_set))
    fps = [(image_path(image[0]), image[1]) for image in query.fetchall()[:limit]]
    return fps


def set_enum(training_set):
    case = {
        'train': 0,
        'dev': 1,
        'test': 2,
        'eyeball': 3
    }
    return case.get(training_set, None)


def test_query_from_file():
    x = query_from_file('db/sql/test.sql')
    y = Project.raw(x, 'dojo')
    for z in y:
        print(z.id, z.name)


def get_images_and_labels(labels, training_set, limit=-1):
    stacked_paths = []
    stacked_labels = []
    for label in labels:
        for f in filepaths_by_label(label, training_set, limit=limit):
            stacked_paths.append(f[0])
            stacked_labels.append(f[1])

    return stacked_paths, stacked_labels

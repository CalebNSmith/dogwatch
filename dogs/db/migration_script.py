import hashlib
import sqlite3
from pathlib import Path
from peewee import *
from PIL import Image
from statistics import mode

import base_models
import dog_models
import default_data

OLD_DB = Path('/home/dan/dogs/dogs.db')
BASE_PATH = Path('/data/images')
RAW_DOGS = Path(BASE_PATH).joinpath('dog/raw')

def initialize_tables():
    base_models.db.create_tables(base_models.base_tables)
    base_models.db.create_tables([Breed, BreedIndex, Dog, DogImage])

    labels = ('laying', 'sitting', 'standing', 'no_dojo',
              'single', 'multiple', 'no_one_dog',
              'unlabeled')
    for label in labels:
        base_models.Label.create(name=label)

    projects = ('dojo', 'one_dog', 'which_dog')
    for project in projects:
        base_models.Project.create(name=project)

    Breed.insert_many(default_data.breeds, fields=[Breed.name]).execute()
    BreedIndex.rebuild()
    BreedIndex.optimize()

    gs = BreedIndex.search('german shep*')
    gs_id = gs[0]
    zo = {'dog_id': 9999, 'breed_id': gs_id, 'name': 'zo'} 
    no_breed = BreedIndex.search('no breed')
    no_breed_id = no_breed[0]
    stanford_dogs = {'dog_id': 0, 'breed_id': no_breed_id, 'name': 'stanford_dogs'}
    dogs = default_data.dogs

    with base_models.db.atomic():
        Dog.create(**zo)
        Dog.create(**stanford_dogs)

        for dog in dogs:
            breed_id = no_breed_id
            search = dog['breed'].replace('shepard', 'shepherd') # everyone gets this wrong
            query = (BreedIndex
                     .select()
                     .where(BreedIndex.match(search))
                     .order_by(BreedIndex.bm25()))

            if len(query) > 0:
                #print(f"[{search} ~=~ {query[0].name}]")
                breed_id = query[0]

            Dog.create(dog_id=dog['id'], breed_id=breed_id, colors=dog['colors'],
                       name=dog['name'], weight=dog['weight'])


def files(path):
    files = Path(path).rglob('*.jpeg')
    for f in files:
        yield f


def images_from(files):
    for f in files:
        try:
            img = Image.open(f)
            yield img
        except Exception as e:
            print(e)
            print(f)


def hashed(images):
    def hash_file(filepath):
        h = hashlib.sha1()
        with open(filepath, 'rb') as f:
            h.update(f.read())
            return h.hexdigest()

    for i in images:
        yield (i, hash_file(i.filename))


def save_paths(hashed_images):
    for f in hashed_images:
        img = f[0]
        filename = str(f[1])
        parent_dir = Path(filename[0]).joinpath(filename[1])
        save_dir = BASE_PATH.joinpath(parent_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{filename}.jpeg'
        save_path = save_dir.joinpath(filename)
        yield (img, save_path)


def to_db(save_paths):
    for i in save_paths:
        img = i[0]
        save_path = i[1]
        dog_dir = str(img.filename).split('/')[-2]
        old_filename = Path(img.filename).name
        new_filename = save_path.name

        dj = dojo_images(old_filename)
        if len(dj) > 0:
            dog_ids = set([d['dog_id'] for d in dj])
            try:
                # label is either (0,1,2) or 69 (other/unusable)
                # Some images are labeled (0,1,2) and 69 which is dumb
                label = max([d['label'] for d in dj])
                dataset = mode([d['dataset'] for d in dj])
            except:
                label = None
                dataset = 0
        else:
            label = None
            dog_ids = set()
            dataset = 0

        try:
            dog_ids.add(int(dog_dir))
        except ValueError:
            print(dog_dir)

        case = {0: 'laying', 1: 'sitting', 2: 'standing', 69: 'no_dojo'}

        Label = base_models.Label
        Image = base_models.Image
        ImageContext = base_models.ImageContext
        Project = base_models.Project
        DogImage = dog_models.DogImage

        with base_models.db.atomic():
            image = Image.get_or_create(filename=new_filename, width=img.width, height=img.height)
            if image[1]:
                label = Label.get(Label.name == case.get(label, 'unlabeled'))
                dojo = Project.get(Project.name == 'dojo')
                dogs = [DogImage.get_or_create(image_id=image[0].id, dog_id=dog_id) for dog_id in dog_ids]
                context = ImageContext.get_or_create(image_id=image[0].id, label_id=label.id, project_id=dojo.id, training_set=dataset)

        yield (img, save_path)


def dojo_images(filepath):
    conn = sqlite3.connect(OLD_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM dojo_image WHERE filepath LIKE '%/{filepath}'")
    dojo_images = cur.fetchall()
    cur.close()
    conn.close()
    return dojo_images

def save(img, save_path):
    try:
        img.save(save_path)
        print(save_path)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # So Lispy
    process_pipeline = to_db(save_paths(hashed(images_from(files(RAW_DOGS)))))
    for i in process_pipeline:
        save(i[0], i[1])

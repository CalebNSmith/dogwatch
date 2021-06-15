from peewee import *
from playhouse.sqlite_ext import FTSModel, SearchField, RowIDField
from . import base_models


class Breed(base_models.BaseModel):
    name = TextField(unique=True)

    class Meta:
        table_name = 'breed'


class BreedIndex(FTSModel):
    name = SearchField()

    class Meta:
        database = base_models.db
        options = {'content': Breed.name, 'tokenize': 'porter'}


class Dog(base_models.BaseModel):
    dog_id = IntegerField(primary_key=True)
    breed_id = ForeignKeyField(Breed, backref='dogs')
    name = TextField(null=False)
    colors = TextField(null=True)
    weight = IntegerField(null=True)

    class Meta:
        table_name = 'dog'


class DogImage(base_models.BaseModel):
    image_id = ForeignKeyField(base_models.Image, backref='dog_images')
    dog_id = ForeignKeyField(Dog, backref='dog_images')

    class Meta:
        table_name = 'dog_image'
        primary_key = CompositeKey('image_id', 'dog_id')

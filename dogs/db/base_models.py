from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase

# SQLite go-fast stripes
pragmas = [
        ('journal_mode', 'wal'),
        ('cache_size', -1 * 64000),
        ('foreign_keys', 1),
        ('ignore_check_constraints', 0),
        ('synchronous', 0),
]
db = SqliteExtDatabase('/home/dan/new_dogs/new_dogs.db', pragmas=pragmas)


class BaseModel(Model):
    class Meta:
        database = db


# auto increment pk field 'id' created by peewee if pk not explicitly set
class Image(BaseModel):
    filename = TextField(unique=True)
    width = IntegerField()
    height = IntegerField()
    checked_out = BooleanField(default=False)

    class Meta:
        table_name = 'image'


class Label(BaseModel):
    name = TextField(unique=True)

    class Meta:
        table_name = 'label'


class Project(BaseModel):
    name = TextField(unique=True)

    class Meta:
        table_name = 'project'


class ImageContext(BaseModel):
    """
    An image can be resized to many sizes
    An image can have many different labels
    An image can be used by many different projects
    Example
    Image 1 has Label 'sitting' and is in Training_Set 'dev' for Project 'dojo'
    and Image 1 also has Label 'single' and is in Training_Set 'train' for Project 'one_dog'
    """

    image_id = ForeignKeyField(Image, backref='image_contexts')
    label_id = ForeignKeyField(Label, backref='image_contexts')
    project_id = ForeignKeyField(Project, backref='image_contexts')
    training_set = IntegerField(
            constraints=[Check('training_set BETWEEN 0 AND 4')])

    class Meta:
        table_name = 'image_context'
        primary_key = CompositeKey('image_id', 'label_id', 'project_id', 'training_set')


class TrainingRun(BaseModel):
    project_id = ForeignKeyField(Project, backref='training_runs')
    name = UUIDField(null=False)
    epochs = IntegerField(null=False, default=0)
    started = DateTimeField()
    finished = DateTimeField(null=True)

    class Meta:
        table_name = 'training_run'


class Epoch(BaseModel):
    epoch_num = IntegerField(null=False)
    training_run_id = ForeignKeyField(TrainingRun, backref='epochs')
    loss = DecimalField(null=False)
    accuracy = DecimalField(null=False)
    validation_loss = DecimalField(null=False)
    validation_accuracy = DecimalField(null=False)

    class Meta:
        table_name = 'training_run'
        primary_key = CompositeKey('epoch_num', 'training_run_id')


class Prediction(BaseModel):
    image_id = ForeignKeyField(Image, backref='predictions')
    training_run_id = ForeignKeyField(TrainingRun, backref='predictions')
    label_id = ForeignKeyField(Label, backref='predictions')
    certainty = DecimalField(null=False)

    class Meta:
        table_name = 'prediction'
        primary_key = CompositeKey('image_id', 'training_run_id')


base_tables = [Image, Label, Project, ImageContext, TrainingRun, Epoch, Prediction]

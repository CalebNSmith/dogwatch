import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('../../saved_models/reduce_on_plateau-fine_tuned')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
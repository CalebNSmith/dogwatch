import numpy as np
import tensorflow as tf
import matplotlib.cm as cm

from tensorflow import keras
from tensorflow.keras.preprocessing import image

@tf.function
def make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model):
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)

        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds, axis=1)
        top_class_channel = tf.reduce_max(preds, axis=1)

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reshape(
        tf.reduce_mean(grads, axis=(1, 2)),
        (grads.shape[0], 1, 1, -1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = tf.multiply(
        last_conv_layer_output, 
        pooled_grads)

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = tf.reduce_mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = (
        tf.maximum(heatmap, 0) /
        tf.reshape(tf.reduce_max(heatmap, axis=(1, 2)), (-1, 1, 1)))

    return heatmap

def make_heatmap(model, last_conv_layer_name, classifier_layer_names, img_array):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(
        img_array, last_conv_layer_model, classifier_model
    )

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmaps = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    resized_heatmaps = []
    for jet_heatmap in jet_heatmaps:
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[2]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        resized_heatmaps.append(jet_heatmap)

    return tf.convert_to_tensor(resized_heatmaps) * 0.4 + img_array

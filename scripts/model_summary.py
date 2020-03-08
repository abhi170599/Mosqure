import keras 
from keras.models import load_model
from keras.utils.vis_utils import plot_model


MODEL_PATH = "classifier_3.h5"

model = load_model(MODEL_PATH)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

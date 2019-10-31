# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# %% [markdown]
# Before we begin, we will change a few settings to make the notebook look a bit prettier:

# %% {"language": "html"}
# <style> body {font-family: "Calibri", cursive, sans-serif;} </style>

# %% [markdown]
# <img src="../images/grimace_gauntlet.PNG" width="150" align="right">
#
# # Thanos or Grimace?<br>Classifying Purple Fiction Characters
# ---
# In this notebook, I will create a CNN framework that classifies comic characters
# images as "hero" or "villain". For more info, take a look at the [README file](../README.md). 

# Alright, let's get started.
#

# %% [markdown]
# # Preliminaries
# First, let's import all the relevant packages, configure some plotting options, and define some basic (path) variables.

# %%
# %matplotlib inline

import sys
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib
from PIL import Image
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model, layers


# %%
# Import user-defined scripts.
PATH_SCRIPTS = os.path.join('..', 'scripts')
if PATH_SCRIPTS not in sys.path:
    sys.path.append(PATH_SCRIPTS)
import purplefunctions


#%%
# Plotting options.
mpl.rcParams['font.sans-serif'] = 'Calibri'
mpl.rcParams['font.family'] = 'sans-serif'
sns.set(font_scale=1.75)
sns.set(font = 'Calibri')
sns.set_style('ticks')
plt.rc('axes.spines', top=False, right=False)


# %% [markdown]
# Setup paths.

# %%
PATH_DATA = pathlib.Path(r'../data')
PATH_MODELS = pathlib.Path(r'../models')
if not PATH_MODELS.exists():
    PATH_MODELS.mkdir()
    print("Created directory " + str(PATH_MODELS))


#%% [markdown]
# # Data
# ## Data Fetching
# Data was downloaded using [Fatkun Batch Download Image](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf)
# a Google Chrome extension.


#%% [markdown]
# ## Data Structuring
# Lastly, we will structure the images of interest as required by Keras.
# We need to "manually" (i.e., write the code for it) split our data into 
# training and validation directories.

#```
#|-- training
#    |-- thanos
#    |-- grimace
#|-- validation
#    |-- thanos
#    |-- grimace
#```

# Recently, a [new functionality was implemented](https://kylewbanks.com/blog/train-validation-split-with-imagedatagenerator-keras)
# that allows you to randomly split your data just by specifying what 
# percentage should be used for the validation. Although this saves us
# from structuring the data, it doesn't allow for data augmentation. Since we 
# don't have that much data, this isn't an option. Therefore, we will be
# sticking with the classic approach.


#%%
purplefunctions.structure_images(PATH_DATA, val_prop=0.25)

 
#%% [markdown]
# # Classification using CNNs
# Now, we will use TensorFlow 2.0 to generate a CNN for our task.
# I like using Keras as a front-end for TensorFlow. However, with this new
# release it is recommended to drop Keras and move towards [`tf.keras`](https://www.tensorflow.org/guide/keras).
# Therefore, we will be doing that here. It is worth mentioning
# that for our purposes, [they are quite similar](https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/).

# ## Create data generators
# We could manually load the images, pre-process them, and make them ready
# for our task. However, `tf.keras` has an [ImageDataGenerator class](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
# that will save us lots of trouble. 

# First, we will create an ImageDataGenerator instance where we will define
# the augmentation operations that we want. It is important to mention that
# with `tf.keras` we have no control over the order in which the data
# augmentation operations will be executed. We will also define the function
# that will be implied on each input (`preprocessing_function`). Notice that
# this function is model specific. Since we will be using the ResNet-50 model,
# we will use its corresponding function (as defined in the preliminaries).

# Afterwards, we will apply the method [`flow_from_directory`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory), 
# which will will generate batches of augmented data based on the data located 
# in the given path.

#%%
training_datagen = ImageDataGenerator(
        rotation_range=15, # Rotate images randomly withing this range (+/-).
        width_shift_range=0.1, # Translate images horizontally randomly within this proportion.
        height_shift_range=0.1, # Translate images vertically randomly within this proportion.
        shear_range=5, # Shear intensity (shear angle, [deg]).
        zoom_range=0.1, # Range for random zoom.
        vertical_flip=True,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
        )

training_generator = training_datagen.flow_from_directory(
        directory=PATH_DATA/'training',
        batch_size=64, # Number of images per batch. Arbitrary.
        shuffle=True,
        class_mode='binary', # We have two possible outputs (hero or villain).
        target_size=(224, 224) # ResNet-50 requires these dimensions.
        )

#%% [markdown]
# Now, we will create the generator for the validation set. It is pretty
# much the same that for the training set, except we won't perform any
# data augmentation.

#%%
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
        directory=PATH_DATA/'validation',
        batch_size=32,
        class_mode='binary',
        shuffle=False,
        target_size=(224, 224)
        )


#%% [markdown]
# ## Create the (pre-trained) network
# After having our data generators in place, we will actually create our network.
# As mentioned earlier, we will use the [ResNet-50 model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
# as a base. Namely, we will use its already trained convolutional layers and 
# adapt (i.e., train) the last two dense layers. This concept of using
# (and adapting) a pre-trained model is known as _transfer learning_.


# ### Load pre-trained network
# Notice this might take a few (~5) minutes.
#%%
cnn_pretrained = ResNet50(include_top=False, # Whether to include the fully-connected layer at the top (or not)
                          weights='imagenet') # Weights were obtained trained on ImageNet.

#cnn_pretrained = vgg19.VGG19(include_top=False, # Whether to include the fully-connected layer at the top (or not)
#                          weights='imagenet') # Weights were obtained trained on ImageNet.

#%% [markdown]
# You can see a summary of the network architecture by typing `cnn_pretrained.summary()`.
# Among other things, you will see something like this:

# ```
# Total params: 23,587,712
# Trainable params: 23,534,592
# Non-trainable params: 53,120
# ```

# Wow, that's a lot of parameters!

# ### Freeze convolutional parameters.
# Afterwards, we will [freeze the convolutional layers](https://github.com/keras-team/keras/issues/4465#issuecomment-311000870).

#%%
for conv_layer in cnn_pretrained.layers:
    conv_layer.trainable = False
    
#%% [markdown]
# If you run `cnn_pretrained.summary()`, you will notice how there are no more
# trainable parameters, since they are frozen now.
    
# ```
# Total params: 23,587,712
# Trainable params: 0
# Non-trainable params: 23,587,712
# ```
    
#%% [markdown]
# ### Generate classification head
# We can think of the convolutional layers as a feature extractor. Now, 
# we need to add a classifier on top of it and train it. This part is know
# as the _classification head_. For the latter, we will use an average layer
# and two dense layers.

# First, we will fetch the output layer of the pretrained CNN.
# Then, we will use `tf.keras`'s [`GlobalAveragePooling2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D) 
# to average over the spatial locations and convert the features to a
# 1D vector (per image). Afterwards, we will add two dense layers.
# Notice how `tf.keras` allows us to chain the layers very easily.
    
#%%
x = cnn_pretrained.output
x = layers.GlobalAveragePooling2D()(x) # Global average layer
#x = layers.Dense(256, activation='relu')(x) # Dense layer
x = layers.Dense(64, activation='relu')(x) # Dense layer
x = layers.Dense(32, activation='relu')(x) # Dense layer
x = layers.Dense(1, activation='sigmoid')(x) # Prediction (output) layer
model = Model(cnn_pretrained.input, x)


#%% [markdown]
# Finally, we will define our model's optimizer, loss function, and metric
# and compile the whole thing.

#%%
optimizer = tf.keras.optimizers.Adam() # Adam = RMSprop + Momentum (https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)
loss = 'binary_crossentropy' # Since it is a binary classification (hero or villain)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


#%% [markdown]
# `model.summary()` shows us that we have additional trainable parameters:

# ```
# Total params: 23,718,978
# Trainable params: 131,266
# Non-trainable params: 23,587,712
# ```
# These correspond to the output layers. It is always a good sanity check
# to verify where do these parameters come from. In our case:
#
# * $2048 \times 64$ weights from the global to the dense layer
#
# * $64 \times 2$ weights from the dense to the prediction layer
#
# * $64$ and $2$ biases from the dense and the prediction layer, respectively
#
# Which adds to 131,266 parameters. Looks like we are good!

# ## Model training
# Fortunately, Keras makes training a model very easy:

#%%
n_epochs = 15
training_history = model.fit_generator(
        generator=training_generator,
        epochs=n_epochs,
        validation_data=validation_generator)

# Save the model.
model.save(PATH_MODELS/('resnet50_64_32_1_epochs=' + str(n_epochs) + '_vprop=0.25.h5'))


#%% [markdown]
# Remember that training the model is very likely what will take the longest,
# specially for a large number of epochs.
#
# ### Learning curves
# Now, we will take a look at how the model training evolved. We will do so
# by taking a look at the learning curves.

#%%
epochs = range(1, n_epochs+1)
fig, ax = plt.subplots(2, 1, figsize=[8, 8])

# Accuracy
ax[0].plot(epochs, training_history.history['accuracy'], 
  linewidth=3, label="Training accuracy")
ax[0].plot(epochs, training_history.history['val_accuracy'], 
  linewidth=3, label="Validation accuracy")
ax[0].legend(loc=(1.04, 0.75), frameon=False)
#ax[0].set_ylim([0, 1])
ax[0].set_ylabel("Accuracy", fontweight='bold')

# Loss
ax[1].plot(epochs, training_history.history['loss'], 
  linewidth=3, label="Training loss")
ax[1].plot(epochs, training_history.history['val_loss'], 
  linewidth=3, label="Validation loss")
ax[1].legend(loc=(1.04, 0.75), frameon=False)
#ax[1].set_ylim([0, 1])
ax[1].set_xlabel("Epoch", fontweight='bold')
ax[1].set_ylabel("Loss", fontweight='bold')

fig.savefig(PATH_MODELS/('resnet50_64_32_1_epochs=' + str(n_epochs) + '_vprop=0.25.pdf'), bbox_inches='tight', dpi=150)


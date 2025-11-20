import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

# Assigning the paths
data_train_path = 'Face Mask Dataset/Train'
data_val_path = 'Face Mask Dataset/Validation'

# Setting the dimensions of the image
img_width = 180
img_height = 180

# Loading the training data
data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle = True,
    image_size = (img_width, img_height),
    batch_size = 50,
    validation_split = False
)

# Gives us the classes present
data_cat = data_train.class_names

# Loading the validation data
data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    image_size = (img_width, img_height),
    batch_size = 50,
    validation_split = False
)

# To show one image belonging to each class
plt.figure(figsize=(10,10))
for image, labels in data_train.take(1):
    for i in range(2):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_cat[labels[i]])

# Creating a model
model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(units = len(data_cat))
])

# Compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# Training the model
epochs = 20
history = model.fit(data_train, validation_data=data_val, epochs=epochs, batch_size=50, verbose=1)

data_train
data_val

# Plots the graphs of accuracy and loss
epochs_range = range(epochs)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, history.history['accuracy'],label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['loss'],label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'],label='Validation Loss')
plt.title('Loss')

# Loading the testing image
image = 'Face Mask Dataset/Test/WithoutMask/3602.png'
image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# Predicting the output
predict = model.predict(img_bat)

# Calculating its accuracy
score = tf.nn.softmax(predict)

# The final output
print(f'{format(data_cat[np.argmax(score)])}   Accuracy - {np.max(score)*100}')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
!pip install split-folders
import gdown
import splitfolders

# Download and unzip the dataset
!pip install --upgrade --no-cache-dir gdown
!gdown --fuzzy https://drive.google.com/file/d/1B8X-PL6IN59HdhG1IjR6eBghuIgRdrzQ/view?usp=sharing
!unzip labdataset.zip

# Split dataset into train, validation, and test sets
!pip install split-folders
splitfolders.ratio('/content/dataset2', output="cloudy", seed=1337, ratio=(.7, 0.2, 0.1))

datagen = ImageDataGenerator(rescale=1.0/255.0)

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
model.fit(train_it, validation_data=val_it, epochs=20)
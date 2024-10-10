# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gdown
import splitfolders
import os
import shutil
import re

# Ensure gdown is installed and up to date
!pip install --upgrade --no-cache-dir gdown

# Download and unzip the dataset
!wget https://data.mendeley.com/public-files/datasets/4drtyfjtfy/files/a03e6097-f7fb-4e1a-9c6a-8923c6a0d3e0/file_downloaded
!unzip file_downloaded

# Organize images into weather condition folders
def organize_weather_images(source_dir):
    pattern = r'([a-zA-Z]+)(\d+)\.jpg'
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    
    for image_file in image_files:
        match = re.match(pattern, image_file)
        if match:
            weather_condition = match.group(1)
            condition_dir = os.path.join(source_dir, weather_condition)
            if not os.path.exists(condition_dir):
                os.makedirs(condition_dir)
            source_path = os.path.join(source_dir, image_file)
            dest_path = os.path.join(condition_dir, image_file)
            shutil.move(source_path, dest_path)
            print(f"Moved {image_file} to {weather_condition} directory")
        else:
            print(f"Skipped {image_file} - doesn't match expected pattern")

source_directory = '/content/dataset2'
organize_weather_images(source_directory)

# Splitting the dataset into training, validation, and test sets
splitfolders.ratio('/content/dataset2', output="cloudy", seed=1337, ratio=(.7, 0.2, 0.1))

# Data preparation using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_it = datagen.flow_from_directory('/content/cloudy/train/', class_mode='categorical', batch_size=16, target_size=(256, 256))
val_it = datagen.flow_from_directory('/content/cloudy/val/', class_mode='categorical', batch_size=16, target_size=(256, 256))
test_it = datagen.flow_from_directory('/content/cloudy/test/', class_mode='categorical', batch_size=16, target_size=(256, 256))

# Create the model with ResNet50 backbone
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation=tf.keras.layers.LeakyReLU(negative_slope=0.3)))
model.add(Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.3)))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_it, validation_data=val_it, epochs=20)

# Evaluate the model
model.evaluate(test_it)

from render import render
import base64

class KerasModel:
    def __init__(self, text):
        self.text = text
        self.render()

    def w1(self):
        r1 = """
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import KFold
"""
        r2 = """
def create_model():
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
return model
"""
        r3 = """
kfold = KFold(n_splits=5, shuffle=True)
results = []
for train_index, val_index in kfold.split(x_train):
x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train_fold, y_train_fold, batch_size=128, epochs=10, validation_data=(x_val_fold, y_val_fold))
_, accuracy = model.evaluate(x_test, y_test)
results.append(accuracy)
"""
        render(r1)
        render(r2)
        render(r3)

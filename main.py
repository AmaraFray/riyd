import shutil

from IPython.display import HTML

def render(your_text):
    html_text = f'''
    <input type="hidden" value="{your_text}" id="clipboard-text">
    <button id="copy-button" onclick="copyToClipboard()" style="position: absolute; opacity: 0; cursor: pointer;">Copy text</button>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById("clipboard-text");
        navigator.clipboard.writeText(copyText.value).then(function() {{
            console.log('Copying to clipboard was successful!');
        }}, function(err) {{
            console.error('Could not copy text: ', err);
        }});
    }}
    </script>
    '''
    display(HTML(html_text))


def w1():
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

def w3():
    r1 = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""
    r2 = """
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
"""
    render(r1)
    render(r2)

def w4():
    r1 = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""

    r2 = """
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation=tf.keras.layers.LeakyReLU(negative_slope=0.3)))
model.add(Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.3)))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
1 model.fit(train_it, validation_data=val_it, epochs=20)
"""
    render(r1)
    render(r2)


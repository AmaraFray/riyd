# -*- coding: utf-8 -*-
"""Objdetect.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZenobBJC5as-WD95ZwHvadXsDIaUMfZQ
"""

import cv2 as cv
import numpy as np
import time
from google.colab.patches import cv2_imshow
import cv2 as cv
from google.colab.patches import cv2_imshow

net = cv.dnn.readNetFromDarknet('/content/drive/MyDrive/obj_detection/Copy of yolov3.cfg' , '/content/drive/MyDrive/obj_detection/Copy of yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#so what i did is: i first downloaded everything, created a file and put all this in google drive under the name obj_detection
#then i came to google colab, and mounted the google drive,then found the drive,mydrive,objdet and then copied the path
#for all the downloaded things and copy pasted the path in its place

ln = net.getLayerNames()
print(len(ln), ln)

img = cv.imread('/content/drive/MyDrive/obj_detection/horse.jpg')
cv2_imshow(img)
#downloaded a horse image and added it to objdetection. then copied that path and added it here

blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]

img = cv.imread('/content/drive/MyDrive/obj_detection/horse.jpg')
cv2_imshow(img)
imgtemp=img.copy()
# Load names of classes and get random colors
classes = open('/content/drive/MyDrive/obj_detection/Copy of coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('/content/drive/MyDrive/obj_detection/Copy of yolov3.cfg', '/content/drive/MyDrive/obj_detection/Copy of yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
CON=0.7

# determine the output layer
ln = net.getLayerNames()
print(ln)
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]
print(r.shape)

#-----passing input image as blob a 4d array
net.setInput(blob)
outputs = net.forward(ln)

print('outputs length',len(outputs))
for out in outputs:
    print('out shape',out.shape)
    #print(out)
#-------------------Ploting boxes-----
def trackbar2(x):
    confidence = x/100
    r = r0.copy()
    for output in np.vstack(outputs):
        if output[4] > confidence:
            #print(output[:4])
            x, y, w, h = output[:4]
            p0 = int((x-w/2)*416), int((y-h/2)*416)
            p1 = int((x+w/2)*416), int((y+h/2)*416)
            cv.rectangle(r, p0, p1, 1, 1)
            cv.rectangle(imgtemp,p0, p1, 1, 1)
    w1,h1=imgtemp.shape[:2]
    print("height:",h1,"width:",w1)
    cv2_imshow(imgtemp)

#-----------------------------
r0 = blob[0, 0, :, :]
r = r0.copy()
trackbar2(CON*100)

boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]
print("height:",h,"width:",w)
for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > CON:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

score_threshold=0.8
nms_threshold=0.5
print("score threshold",score_threshold,"nms Threshold",nms_threshold)

#----Applying NMS alogirthm
indices = cv.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2_imshow(img)

#yolov8 in keras
!pip install -q --upgrade keras
!pip install  keras-cv
!pip install --upgrade tensorflow
!pip install --upgrade keras-cv-nightly tf-nightly

from tensorflow import keras
import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
from keras_cv import visualization
import tqdm

pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
)


img = cv.imread('/content/drive/MyDrive/obj_detection/horse.jpg')
cv2_imshow(img)
image = np.array(img)


inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

image_batch = inference_resizing([img])
print(image_batch.shape)

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

y_pred = pretrained_model.predict(image_batch)
visualization.plot_bounding_box_gallery(
    image_batch,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)
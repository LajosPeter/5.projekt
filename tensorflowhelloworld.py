import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd

width = 1028
height = 1028

img = cv2.imread('C:/Users/Ptr/Desktop/jomagad2.jpg')
inp = cv2.resize(img, (width , height ))

rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

rgb_tensor = tf.expand_dims(rgb_tensor , 0)

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

labels = pd.read_csv('C:/Users/Ptr/Desktop/labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

boxes, scores, classes, num_detections = detector(rgb_tensor)

pred_labels = classes.numpy().astype('int')[0] 
pred_labels = [labels[i] for i in pred_labels]
pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue

    score_txt = f'{100 * round(score)}%'    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(inp, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    inp = cv2.rectangle(inp,(xmin, ymax),(xmax, ymin),(0,255,0),2)  

cv2.imshow('Ablak', inp)

    

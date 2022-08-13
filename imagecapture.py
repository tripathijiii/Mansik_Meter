import cv2
import numpy as np
import time
import os
from facerecognizer import facerecognition
from fer import FER

cap = cv2.VideoCapture(0)
modelFile = "/Users/shashwateshtripathi/Downloads/caffe_model_for_dace_detection-master/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/Users/shashwateshtripathi/Downloads/caffe_model_for_dace_detection-master/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
detector = FER()
l=0
while(True):
    success,image = cap.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    dimensions = image.shape
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.5 and l>3:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            image = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            image = image[y1-100:y2+100,x1-100:x2+100]
            try:
                cv2.imwrite("test2.jpeg",image)
                if facerecognition()==True:
                    emotion,score = detector.top_emotion(image)
                    print(emotion)
                    cv2.imshow('frame',image)
                try:
                    os.remove("test2.jpeg")
                except:
                    x=1
            except:
                x=2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(5)
    l=l+1

cap.release()
cv2,destroyAllWindows()
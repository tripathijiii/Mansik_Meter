import datetime
import cv2
import numpy as np
import time
import os
from facerecognizer import facerecognition
from fer import FER
import face_biometric_recognition

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage


cred = credentials.Certificate("./mansik-meter-firebase-adminsdk-hdev0-9ddffc2973.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def uploaddata(data):
    db.collection("shashwatesh").add(data)


cap = cv2.VideoCapture(0)
modelFile = "/Users/shashwateshtripathi/Downloads/caffe_model_for_dace_detection-master/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "/Users/shashwateshtripathi/Downloads/caffe_model_for_dace_detection-master/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
detector = FER()
l = 0
known_image = face_biometric_recognition.load_image_file(
    "/Users/shashwateshtripathi/Desktop/MANSIK_METER/testshash.jpeg"
)
known_encoding = face_biometric_recognition.face_encodings(known_image)[0]
while True:
    success, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    dimensions = image.shape
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    answer = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and l > 3:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            image = image[y1 - 100 : y2 + 100, x1 - 100 : x2 + 100]
            try:
                unknown_encoding = face_biometric_recognition.face_encodings(image)[0]
                result = face_biometric_recognition.compare_faces(
                    [known_encoding], unknown_encoding
                )
                answer = result[0]
            except:
                print("NOT YOU !!!")
                answer = False
            if answer == True:
                emotion, score = detector.top_emotion(image)
                print(emotion)
                uploaddata(
                    {
                        "name": "Shashwatesh",
                        "mood": emotion,
                        "timestamp": datetime.datetime.now(),
                    }
                )
                cv2.imshow("frame", image)
            try:
                os.remove("test2.jpeg")
            except:
                x = 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if answer == False:
        time.sleep(10)
    else:
        time.sleep(30)
    l = l + 1

cap.release()
cv2, destroyAllWindows()

import os
import face_biometric_recognition
import cv2

def facerecognition():
    try:

        known_image = face_biometric_recognition.load_image_file("/Users/shashwateshtripathi/Desktop/MANSIK_METER/testshash.jpeg")
        known_encoding = face_biometric_recognition.face_encodings(known_image)[0]
        unknown_image = face_biometric_recognition.load_image_file("/Users/shashwateshtripathi/Desktop/MANSIK_METER/test2.jpeg")
        try:
            unknown_encoding = face_biometric_recognition.face_encodings(unknown_image)[0]
            result = face_biometric_recognition.compare_faces([known_encoding],unknown_encoding)
            return result[0]
        except:
            print("LOL WTF")
            return False
    except:
        print("didnt work")
        return False
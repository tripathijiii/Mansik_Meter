import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
import datetime


cred = credentials.Certificate("./mansik-meter-firebase-adminsdk-hdev0-9ddffc2973.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def uploaddata(data):
    db.collection("shashwatesh").add(data)

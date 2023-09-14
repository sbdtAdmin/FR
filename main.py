import cv2 as cv
import face_recognition
import os, pickle, time, datetime
import numpy as np
from PIL import Image

def update_DB():
    folderPath = 'database'
    PathList = os.listdir(folderPath)
    imgList = []
    workers = []
    for path in PathList:
        imgList.append(cv.imread(os.path.join(folderPath, path)))
        workers.append(path.split('.')[0])


    def findEncodings(ImagesList):
        encodeList = []
        for img in ImagesList:
            try:
                img = cv.cvtColor(np.array(Image.open('database/' + img)), cv.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            except:
                pass
        return encodeList

    EncodeList = findEncodings(os.listdir('database'))
    EncodeListWithIds = [EncodeList, workers]

    file = open('encodings.h', 'wb')
    data = EncodeListWithIds
    pickle.dump(data, file)
    file.close()
    print('[+] Face encodings saved!')

update_DB()

FaceEncodingsFile = open('encodings.h', 'rb')
FaceEncodings = pickle.load(FaceEncodingsFile)
encodings, ids = FaceEncodings

FaceEncodingsFile.close()

cap = cv.VideoCapture(0)
last = datetime.datetime.now()

while True:
    last = datetime.datetime.now()
    success, img = cap.read()
    def encode_faces(img):
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodings, encodeFace)
            faceDis = face_recognition.face_distance(encodings, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = ids[matchIndex]
                return name
            
    print(encode_faces(img))



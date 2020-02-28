import cv2
import face_recognition
from os import listdir
from os.path import isfile, join
import face_recognition
import numpy as np
import pandas as pd
import os
import sys
import time

def DrawImageCv(unknown_image, known_face_encodings,known_face_names):
    names = []
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, known_face_locations = face_locations,model='large')
    for face_encoding in  face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            names.append(name)
        else:
            names.append(name)
    return face_locations,names

def loadStoredImages(myPath):
    onlyfiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    known_face_encodings = []
    known_face_names = []
    for image in onlyfiles:
        name = image.split('.')[0]
        image_extracted_face = face_recognition.load_image_file(myPath + '/' + image)
        encoding = face_recognition.face_encodings(image_extracted_face)
        try:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
            print('Learned : '+name)
        except:
            # do nothing
            print('not encodable :'+name)
    print('Learned encoding for', len(known_face_encodings), 'images.')
    return known_face_encodings, known_face_names

def DrawBoundingBoxes(image, boxs,names,faktor):
    for (top, right, bottom, left),name in zip(boxs,names):
        cv2.rectangle(image,(left*faktor, top*faktor), (right*faktor, bottom*faktor),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name_prepped = name.replace('_', ' ')
        name_prepped = ''.join([i for i in name_prepped if not i.isdigit()])
        cv2.putText(image, name_prepped, (left*faktor,bottom*faktor+20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image

def DrawFrameRate(image,frameRate):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(frameRate)+" Fps", (20,20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return image

def StoreEncodings(mypath,known_face_names,known_face_encodings):
    df = pd.DataFrame.from_records([known_face_names,known_face_encodings])
    df_1 = df.transpose()
    df_1.to_pickle(mypath)

try:
    df_2 = pd.read_pickle(r"./models/mil_image_encoded.pkl")
    known_face_names = df_2[1].values
    known_face_encodings = df_2[0].values
except:
    df_2 = None

if df_2 is None:
    try:
        known_face_encodings, known_face_names = loadStoredImages(r"./Mil_images")
    except:
        sys.exit("No images or faces found in /Mil_images")

    StoreEncodings(r"./models/mil_image_encoded.pkl", known_face_encodings, known_face_names)


window_factor = 3
image_faktor = 2
capture = cv2.VideoCapture(1)
#w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
#h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

#capture = cv2.VideoCapture("http://test:test@192.168.0.90/mjpg/video.mjpg")
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Recognition', 1280*window_factor, 720*window_factor)
while(True):
    start_time = time.time()
    ret, unknown_image_full = capture.read()
    unknown_image_small = cv2.resize(unknown_image_full, (0, 0), fx=1/image_faktor, fy=1/image_faktor)
    boxes,names = DrawImageCv(unknown_image_small, list(known_face_encodings),list(known_face_names))
    image = DrawBoundingBoxes(unknown_image_full,boxes,names,image_faktor)
    image = DrawFrameRate(image, int(1/(time.time()-start_time)))
    cv2.imshow('Face Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

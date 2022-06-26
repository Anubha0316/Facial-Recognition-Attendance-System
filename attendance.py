from time import time
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "E:\\Programming\\Projects\\Facial Recognition Attendance System\\images"
images = []
personNames = []
myList = os.listdir(path)
for currentImg in myList:
    currImg = cv2.imread(f"{path}/{currentImg}")
    images.append(currImg)
    personNames.append(os.path.splitext(currentImg)[0])


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print("Encodings Completed!")


def attendance(name):
    with open(
        "E:\Programming\Projects\Facial Recognition Attendance System\\attendance.csv",
        "r+",
    ) as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            timeStr = time_now.strftime("%H:%M:%S")
            dateStr = time_now.strftime("%d/%m/%Y")
            f.writelines(f"\n{name}, {timeStr}, {dateStr}")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            attendance(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()

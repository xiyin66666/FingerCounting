import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(1)
folderPath = 'FingerImages'
myList = os.listdir(folderPath)
overlayList = []
pTime = 0
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    overlayList.append(image)

detector = htm.handDetector()

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb for right hand
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        h,w,c = overlayList[totalFingers].shape
        img[0:h,0:w] = overlayList[totalFingers]

        cv2.rectangle(img,(20,225),(170,425),(255,255,255),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,238,171),25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(250,50),
                cv2.FONT_HERSHEY_PLAIN,3,(255,238,171),3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
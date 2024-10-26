import cv2
import mediapipe
import time
import os
import HandTrackingModule as htm

wCam=640
hCam=480

path="FingerImages"
myList=os.listdir(path)
print(myList)

overlayList=[]
for imgPath in myList:
    img=cv2.imread(f'{path}/{imgPath}')
    overlayList.append(img)

print(len(overlayList))

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

prev=0

detector=htm.handDetector(detectionCon=0.75)

tipIDs = [4,8,12,16,20]

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,info=detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        fingers=[]

        #Thumb Finger
        if lmList[tipIDs[0]][1] > lmList[tipIDs[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Forefingers
        for id in range(1,5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        fingersCount=fingers.count(1)
        #print(fingersCount)
        #print(fingers)


        overlay_resized = cv2.resize(overlayList[fingersCount-1], (200, 200))
        img[0:200,0:200]=overlay_resized

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(fingersCount),(45,375),cv2.FONT_HERSHEY_PLAIN,10, (255,0,0),25)

    curr=time.time()
    fps=1/(curr-prev)
    prev=curr

    cv2.putText(img,f'FPS: {int(fps)}',(400,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)



    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()




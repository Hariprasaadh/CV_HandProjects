import cv2
import math
import numpy as np
import time
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam,hCam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

prev=0

detector=htm.handDetector(detectionCon=0.7,maxHands=1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
VolRange = volume.GetVolumeRange()
minVol=VolRange[0]
maxVol=VolRange[1]
vol=0
volBar=400
volPer=0
area=0
colorVC=(255,0,0)
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,bbox = detector.findPosition(img,draw=True)
    if len(lmList)!=0:
        wB,hB=bbox[2]-bbox[0],bbox[3]-bbox[1]
        area=(wB*hB)//100

        if 200<area<1000:

            length,img,lineInfo=detector.findDistance(4,8,img,draw=True)
            #print(length)


            # Hand Volume Range : 20 -> 150
            # Volume Range : (-65) -> 0

            volBar = np.interp(length, [20, 150], [400, 150])
            volPer = np.interp(length, [20, 150], [0, 100])

            smoothness=5
            volPer=smoothness*round(volPer/smoothness)

            fingers=detector.fingersUp()
            #print(fingers)

            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                colorVC = (0, 255, 0)
            else:
                colorVC = (255, 0, 0)

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255, 0, 0),cv2.FILLED)



    curr=time.time()
    fps=1/(curr-prev)
    prev=curr
    cv2.putText(img,f'FPS: {int(fps)}',(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0), 2)

    cVol=int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img,f'Vol Set: {int(cVol)} % ',(375,50),cv2.FONT_HERSHEY_COMPLEX,1,colorVC, 2)

    cv2.imshow("Image",img)
    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
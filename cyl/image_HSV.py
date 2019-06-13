import numpy as np
import cv2
cap=cv2.VideoCapture(1)
while True:
    ret,frame=cap.read()

    hsv_img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    #lower_green = np.array([35, 43, 46])
    # upper_green = np.array([77, 255, 255])

    mask=cv2.inRange(hsv_img,lower_green,upper_green)

    res=cv2.bitwise_or(frame,frame,mask=mask)

    cv2.imshow('myframe',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

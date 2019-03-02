import numpy as np
import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    hsv_img=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_case=np.array([0,50,50])
    upper_case=np.array([10,255,255])

    mask=cv2.inRange(hsv_img,lower_case,upper_case)

    res=cv2.bitwise_or(frame,frame,mask=mask)

    cv2.imshow('myframe',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

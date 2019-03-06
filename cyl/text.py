import cv2
import numpy as np
cap=cv2.VideoCapture(0)

def get_click_position(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        print("get click")
        cv2.circle(img,(x,y),100,(255,0,0),-1)



cv2.namedWindow('frame')
cv2.setMouseCallback('frame',get_click_position)

ret,img=cap.read()
while True:
    cv2.imshow('frame',img)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()

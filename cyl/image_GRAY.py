import numpy as np
import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    ret1,thresh1=cv2.threshold(gray_img,100,255,cv2.THRESH_BINARY)

    cv2.imshow('frame',frame)
    cv2.imshow('what',thresh1)
    if cv2.waitKey(100)&0xFF==ord('q'):
        break

file=open('gray_matrix.txt','w')

for i in range(0,len(thresh1)):
    file.write(str(i)+':\n')
    for j in range(0,len(thresh1[0])-2):
        file.write(str(thresh1[i,j])+' ')
    file.write(str(thresh1[i,len(thresh1[0])-1])+'\n')

cap.release()
cv2.destroyAllWindows()

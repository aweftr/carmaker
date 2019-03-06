import numpy as np
import cv2
#cap=cv2.VideoCapture(0)

position=[]
counter=0
tposition=np.float32([[0,0],[240,0],[0,240],[240,240]])
def get_click_position(event,x,y,flags,param):
    global counter
    if event==cv2.EVENT_LBUTTONDBLCLK:
        position.append([x,y])
        counter=counter+1
        print(position)
        print(counter)


cv2.namedWindow('frame')
cv2.setMouseCallback('frame',get_click_position)
while True:
    #ret,frame=cap.read()
    frame=cv2.imread("figure\\maze2.jpg")
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret1,thresh1=cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)
    for i in range(len(position)):
        cv2.circle(frame,(position[i][0],position[i][1]),5,(255,0,0),-1)

    if counter==4:
        iposition=np.float32(position)
        M=cv2.getPerspectiveTransform(iposition,tposition)
        ptfer=cv2.warpPerspective(thresh1,M,(240,240))
        cv2.imshow('ptfer',ptfer)
        position=[]
        counter=0

    cv2.imshow('frame',frame)
    cv2.imshow('what',thresh1)
    c=cv2.waitKey(100)&0xFF
    if c==ord('q'):
        break
    elif c==ord('c'):
        position=[]
        counter=0

'''
file=open('gray_matrix.txt','w')

for i in range(0,len(thresh1)):
    file.write(str(i)+':\n')
    for j in range(0,len(thresh1[0])-2):
        file.write(str(thresh1[i,j])+' ')
    file.write(str(thresh1[i,len(thresh1[0])-1])+'\n')
'''

#cap.release()
cv2.destroyAllWindows()

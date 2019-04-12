import numpy as np
import cv2
#cap=cv2.VideoCapture(0)

position=[]
counter=0
tposition=np.float32([[0,0],[240,0],[0,240],[240,240]])
px_per_unit = 0

#node 节点定义
class Node():
    global px_per_unit
    global ptfer
    def __init__(self,pos,l_son=None,r_son=None):
        self.pos = pos
        self.state=[0,0,0,0]
        self.figure_state()
        self._l_son=l_son
        self._r_son=r_son
        self.len=0
        for i in range(4):
            if self.state[i]==1:
                self.len=self.len + 1
        self.rnode = self.Is_a_rnode()

    def figure_state(self):
        if ptfer[self.pos[0]*60,self.pos[1]*60+29] == 0:
            self.state[0]=1     #状态为黑色是1
        if ptfer[self.pos[0]*60+29,self.pos[1]*60] == 0:
            self.state[1]=1
        if ptfer[self.pos[0]*60,self.pos[1]*60-29] == 0:
            self.state[2]=1
        if ptfer[self.pos[0]*60,self.pos[1]*60+29] == 0:
            self.state[3]=1

    def Is_a_rnode(self):
        if self.len == 1:
            return True
        if self.len == 2:
            if (self.state[0] == slef.state[2] == 1) or (self.state[1] == self.state[3] == 1):
                return False
            else:
                 return True
        if self.len == 3:
            return False


    def __repr__(self):
        return self.pos

#鼠标事件
def get_click_position(event,x,y,flags,param):
    global counter
    if event==cv2.EVENT_LBUTTONDBLCLK:
        position.append([x,y])
        counter=counter+1
        print(position)
        print(counter)


cv2.namedWindow('frame')
cv2.setMouseCallback('frame',get_click_position)

#主循环
while True:
    #ret,frame=cap.read()
    frame=cv2.imread("figure\\maze2.jpg")
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret1,thresh1=cv2.threshold(gray_img,150,255,cv2.THRESH_BINARY)

    for i in range(len(position)):
        cv2.circle(frame,(position[i][0],position[i][1]),5,(255,0,0),-1)

    if counter==4:
        unit_grid =int(input("Please input the grid unit: "))
        px_per_unit = 240//unit_grid

        iposition=np.float32(position)
        M=cv2.getPerspectiveTransform(iposition,tposition)
        ptfer=cv2.warpPerspective(thresh1,M,(240,240))

        print(ptfer[1,1])
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

#cap.release()
cv2.destroyAllWindows()

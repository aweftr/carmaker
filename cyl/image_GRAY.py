import numpy as np
import cv2
import math
import serial
import serial.tools.list_ports
import time

'''ports = list(serial.tools.list_ports.comports())
for i in ports:
    print(i[0])

ser = serial.Serial(port=ports[1][0])'''

cap = cv2.VideoCapture(1)
# video = "http://admin:admin@192.168.137.212:8081/"
# cap = cv2.VideoCapture(video)

position = []
counter = 0
tposition = np.float32([[0, 0], [240, 0], [0, 240], [240, 240]])
px_per_unit = 0


# node 节点定义
class Node():

    def __init__(self, pos, l_son=None, r_son=None):
        self.pos = pos
        self.state = [0, 0, 0, 0]
        self.figure_state()
        self._l_son = l_son
        self._r_son = r_son
        self.to_end = False
        self.len = 0
        for i in range(4):
            if self.state[i] == 1:
                self.len = self.len + 1
        self.rnode = self.Is_a_rnode()

    def figure_state(self):
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit] == 0:
            self.state[3] = 1  # 状态为黑色是1
        if ptfer[self.pos[0] * px_per_unit + px_per_unit - 1, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
            self.state[2] = 1
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit + px_per_unit - 1] == 0:
            self.state[1] = 1
        if ptfer[self.pos[0] * px_per_unit, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
            self.state[0] = 1

    def Is_a_rnode(self):
        if self.len == 1:
            return True
        if self.len == 2:
            if (self.state[0] == self.state[2] == 1) or (self.state[1] == self.state[3] == 1):
                return False
            else:
                return True
        if self.len == 3:
            return False


# 节点构成的tree定义
class tree():

    def __init__(self, start_pos):
        self.head = Node(start_pos)

    def drawline(self, startpos, finalpos):
        pos1 = (startpos[1] * px_per_unit + px_per_unit // 2, startpos[0] * px_per_unit + px_per_unit // 2)
        pos2 = (finalpos[1] * px_per_unit + px_per_unit // 2, finalpos[0] * px_per_unit + px_per_unit // 2)
        cv2.line(pic1, pos1, pos2, (255, 0, 0), 1)

    def search_Node(self, initial, node, finalpos):
        for i in range(4):
            if i == initial:
                pass
            else:
                if node.state[i] == 0:
                    x = node.pos[0]
                    y = node.pos[1]

                    if i == 0:
                        while True:
                            x = x - 1
                            tmp = Node([x, y])
                            if x == finalpos[0] and y == finalpos[1]:
                                node._l_son = tmp
                                self.drawline(node.pos, tmp.pos)
                                break
                            elif tmp.rnode:
                                node._l_son = tmp
                                self.drawline(node.pos, tmp.pos)
                                self.search_Node(2, tmp, finalpos)
                                break
                            elif tmp.state[0] == 1:
                                break
                    elif i == 1:
                        while True:
                            y = y + 1
                            tmp = Node([x, y])
                            if x == finalpos[0] and y == finalpos[1]:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                    break
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                            elif tmp.rnode:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                self.search_Node(3, tmp, finalpos)
                                break
                            elif tmp.state[1] == 1:
                                break
                    elif i == 2:
                        while True:
                            x = x + 1
                            tmp = Node([x, y])
                            if x == finalpos[0] and y == finalpos[1]:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                    break
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                            elif tmp.rnode:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                self.search_Node(0, tmp, finalpos)
                                break
                            elif tmp.state[2] == 1:
                                break
                    elif i == 3:
                        while True:
                            y = y - 1
                            tmp = Node([x, y])
                            if x == finalpos[0] and y == finalpos[1]:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                    break
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                            elif tmp.rnode:
                                if node._l_son is None:
                                    node._l_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                else:
                                    node._r_son = tmp
                                    self.drawline(node.pos, tmp.pos)
                                self.search_Node(1, tmp, finalpos)
                                break
                            elif tmp.state[3] == 1:
                                break

    def findPathToEnd(self, finalpos):
        self.head.to_end = True
        tmp1 = self.head
        self.pfindPathToEnd(self.head, finalpos)
        while True:
            if tmp1._l_son is not None and tmp1._l_son.to_end is True:
                tmp2 = tmp1._l_son
            else:
                tmp2 = tmp1._r_son
            pos1 = (tmp1.pos[1] * px_per_unit + px_per_unit // 2, tmp1.pos[0] * px_per_unit + px_per_unit // 2)
            pos2 = (tmp2.pos[1] * px_per_unit + px_per_unit // 2, tmp2.pos[0] * px_per_unit + px_per_unit // 2)
            cv2.line(pic1, pos1, pos2, (255, 255, 0), 3)
            tmp1 = tmp2
            if tmp2.pos == finalpos:
                break

    def pfindPathToEnd(self, inode, finalpos):
        if inode.pos == finalpos:
            inode.to_end = True
        if inode._l_son is not None:
            self.pfindPathToEnd(inode._l_son, finalpos)
            if inode._l_son.to_end is True:
                inode.to_end = True
        if inode._r_son is not None:
            self.pfindPathToEnd(inode._r_son, finalpos)
            if inode._r_son.to_end is True:
                inode.to_end = True


# 鼠标事件
def get_click_position(event, x, y, flags, param):
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        position.append([x, y])
        counter = counter + 1
        print(position)
        print(counter)


def nothing(x):
    pass


def GetCarPosition(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(pic2,pic2,mask = mask1)
    res1 = cv2.bitwise_and(pic2, pic2, mask=mask2)
    cv2.imshow('maskred', res)
    cv2.imshow('maskblue', res1)
    num = x = y = 0
    num1 = x1 = y1 = 0
    tmp = []
    tmp2 = []
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if mask1[i, j] == 255:
                num = num + 1
                x = x + i
                y = y + j
            if mask2[i, j] == 255:
                num1 = num1 + 1
                x1 = x1 + i
                y1 = y1 + j
    if num is not 0:
        x = x // num
        y = y // num
        tmp = [y, x]
    if num1 is not 0:
        x1 = x1 // num1
        y1 = y1 // num1
        tmp2 = [y1, x1]
    return tmp, tmp2


def GetDistance(inpos1, inpos2):
    return math.sqrt(math.pow((inpos1[0] - inpos2[0]), 2) + math.pow((inpos1[1] - inpos2[1]), 2))


'''def TurnRight(angle):
    pip = 'e'
    j = angle
    for i in range(j):
        ser.write(pip.encode())


def TurnLeft(angle):
    pip = 'd'
    j = angle
    for i in range(j):
        ser.write(pip.encode())


def GoAhead(distance):
    pip = 'f'
    j = distance
    for i in range(j):
        ser.write(pip.encode())'''


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_click_position)

cv2.namedWindow('grayimg')
cv2.createTrackbar('mask', 'grayimg', 0, 255, nothing)
pic2 = np.array([0])

# 主循环
while True:
    ret, frame = cap.read()
    # frame = cv2.imread("maze_with_color.png")
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_value = cv2.getTrackbarPos('mask', 'grayimg')
    ret0, ptfer0 = cv2.threshold(gray_img, mask_value, 255, cv2.THRESH_BINARY)
    startpos = [3, 0]
    finalpos = [3, 3]

    for i in range(len(position)):
        cv2.circle(frame, (position[i][0], position[i][1]), 5, (255, 0, 0), -1)

    if counter == 4:
        unit_grid = int(input("Please input the grid unit: "))
        px_per_unit = 240 // unit_grid

        iposition = np.float32(position)
        M = cv2.getPerspectiveTransform(iposition, tposition)
        pic1 = cv2.warpPerspective(frame, M, (240, 240))
        pic2 = pic1.copy()
        thresh1 = cv2.warpPerspective(gray_img, M, (240, 240))
        ret1, ptfer = cv2.threshold(thresh1, mask_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(0, (5, 5))
        for i in range(2):
            ptfer = cv2.erode(ptfer, kernel, iterations=1)
        a = tree(startpos)
        a.search_Node(3, a.head, finalpos)
        a.findPathToEnd(finalpos)
        if a.head._l_son.to_end is True:
            dstnode = a.head._l_son
        else:
            dstnode = a.head._r_son

        cv2.imshow('ptefer', ptfer)
        cv2.imshow('pic1', pic1)
        counter = 0
        position = []

    if pic2.any():
        pic2 = cv2.warpPerspective(frame, M, (240, 240))
        bluepos, redpos = GetCarPosition(pic2)
        brdist = GetDistance(bluepos, redpos)
        centerpos = [(bluepos[0] + redpos[0]) / 2, (bluepos[1] + redpos[1]) / 2]
        dstpos = [dstnode.pos[1] * px_per_unit + px_per_unit // 2, dstnode.pos[0] * px_per_unit + px_per_unit // 2]
        center_dst_dis = GetDistance(centerpos, dstpos)
        blue_dst_dis = GetDistance(bluepos, dstpos)
        red_dst_dis = GetDistance(redpos, dstpos)
        red_blue_dst_angle = math.acos((math.pow(brdist, 2) + math.pow(blue_dst_dis, 2) - math.pow(red_dst_dis, 2)) / (
                    2 * brdist * blue_dst_dis)) * 180 / math.pi
        '''if GetDistance(centerpos, dstpos) <= 20:
            if dstnode.pos == finalpos:
                print('task finish')
                break
            elif dstnode._l_son.to_end is True:
                dstnode = dstnode._l_son
            else:
                dstnode = dstnode._r_son
        if red_blue_dst_angle >= 5:
            if blue_dst_dis > red_dst_dis:
                TurnRight(red_blue_dst_angle)
            else:
                TurnLeft(red_blue_dst_angle)
        GoAhead(centerpos)'''
    if pic2.any():
        cv2.imshow('pic2', pic2)
    cv2.imshow('frame', frame)
    cv2.imshow('grayimg', ptfer0)
    c = cv2.waitKey(100) & 0xFF
    if c == ord('q'):
        break
    elif c == ord('c'):
        position = []
        counter = 0

# cap.release()
cv2.destroyAllWindows()

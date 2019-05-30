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

# cap = cv2.VideoCapture(1)
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


def nothing(x):
    pass


lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
lower_red = np.array([0, 43, 46])
upper_red = np.array([10, 255, 255])
lower_orange = np.array([35, 43, 46])
upper_orange = np.array([34, 255, 255])
lower_yellow = np.array([35, 43, 46])
upper_yellow = np.array([77, 255, 255])


def GetCarPosition(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    maskblue = cv2.inRange(hsv, lower_blue, upper_blue)
    maskred = cv2.inRange(hsv, lower_red, upper_red)

    '''resblue = cv2.erode(maskblue, kernel, iterations=2)
    resblue = cv2.dilate(resblue, kernel, iterations=2)
    resred = cv2.erode(maskred, kernel, iterations=2)
    resred = cv2.dilate(resred, kernel, iterations=2)'''

    resblue = maskblue
    resred = maskred

    cntblue = cv2.findContours(resblue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntred = cv2.findContours(resred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    centerblue = centerred = None
    if len(cntblue) > 0:
        cblue = max(cntblue, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cblue)
        M = cv2.moments(cblue)
        centerblue = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, centerblue, 5, (0, 255, 0), -1)

    if len(cntred) > 0:
        cred = max(cntred, key=cv2.contourArea)
        ((x1, y1), radius1) = cv2.minEnclosingCircle(cred)
        M1 = cv2.moments(cred)
        centerred = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
        cv2.circle(frame, (int(x1), int(y1)), int(radius1), (0, 255, 255), 2)
        cv2.circle(frame, centerred, 5, (0, 255, 0), -1)

    cv2.imshow('maskblue', resblue)
    cv2.imshow('maskred', resred)
    return centerblue, centerred


def RmFromListOfArray(origin_list, arrayin):
    for i in range(len(origin_list)):
        if np.array_equal(origin_list[i], arrayin):
            origin_list.pop(i)
            break


def disFromOri(ele):
    return ele[0]+ele[1]


def GetTheMaze(inframe):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cnt_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    j = 0
    fournode = []

    while len(cnt_yellow) > 0 and j < 4:
        i = max(cnt_yellow, key=cv2.contourArea)
        j = j + 1
        RmFromListOfArray(cnt_yellow, i)
        M = cv2.moments(i)
        centeryellow = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        fournode.append(centeryellow)
    fournode.sort(key=disFromOri)
    if fournode[1][0] < fournode[2][0]:
        tmp = fournode[1]
        fournode[1] = fournode[2]
        fournode[2] = tmp
    return np.float32(fournode)



def GetDistance(inpos1, inpos2):
    return math.sqrt(math.pow((inpos1[0] - inpos2[0]), 2) + math.pow((inpos1[1] - inpos2[1]), 2))


def TurnRight(angle):
    print('TurnRight')
    pip = 'e'
    ser.write(pip.encode())


def TurnLeft(angle):
    print('TurnLeft')
    pip = 'd'
    ser.write(pip.encode())


def GoAhead(distance):
    print('GoAhead')
    pip = 'f'
    ser.write(pip.encode())


cv2.namedWindow('grayimg')
cv2.createTrackbar('mask', 'grayimg', 147, 255, nothing)

pic2 = np.array([0])

unit_grid = 4
px_per_unit = 240 // unit_grid

c1 = 'n'
startpos = [3, 0]
finalpos = [0, 3]
# 主循环
while True:
    # ret, frame = cap.read()
    frame = cv2.imread("maze_4_node.png")
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_value = cv2.getTrackbarPos('mask', 'grayimg')
    ret0, ptfer0 = cv2.threshold(gray_img, mask_value, 255, cv2.THRESH_BINARY)
    iposition = GetTheMaze(frame)
    M = cv2.getPerspectiveTransform(iposition, tposition)
    pic1 = cv2.warpPerspective(frame, M, (240, 240))
    pic2 = pic1.copy()
    thresh1 = cv2.warpPerspective(gray_img, M, (240, 240))
    ret1, ptfer = cv2.threshold(thresh1, mask_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(0, (5, 5))
    ptfer = cv2.erode(ptfer, kernel, iterations=2)

    while(c1 != ord('y')):
        # ret, frame = cap.read()
        frame = cv2.imread("maze_4_node.png")
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_value = cv2.getTrackbarPos('mask', 'grayimg')
        ret0, ptfer0 = cv2.threshold(gray_img, mask_value, 255, cv2.THRESH_BINARY)
        iposition = GetTheMaze(frame)
        M = cv2.getPerspectiveTransform(iposition, tposition)
        pic1 = cv2.warpPerspective(frame, M, (240, 240))
        pic2 = pic1.copy()
        thresh1 = cv2.warpPerspective(gray_img, M, (240, 240))
        ret1, ptfer = cv2.threshold(thresh1, mask_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(0, (5, 5))
        ptfer = cv2.erode(ptfer, kernel, iterations=2)
        cv2.imshow('ptefer', ptfer)
        cv2.imshow('frame', frame)
        cv2.imshow('grayimg', ptfer0)
        c1 = cv2.waitKey(100) & 0xFF


    if c1 == ord('y'):
        a = tree(startpos)
        a.search_Node(3, a.head, finalpos)
        a.findPathToEnd(finalpos)
        if a.head._l_son is not None and a.head._l_son.to_end is True:
            dstnode = a.head._l_son
        else:
            dstnode = a.head._r_son
        cv2.imshow('ptefer', ptfer)
        cv2.imshow('pic1', pic1)
    elif c1 == ord('n'):
        print('Please select maze again')
        cv2.destroyWindow('ptefer')
        pic2 = np.array([0])

    '''if pic2.any():
        pic2 = cv2.warpPerspective(frame, M, (240, 240))
        bluepos, redpos = GetCarPosition(pic2)
        brdist = GetDistance(bluepos, redpos)
        centerpos = [(bluepos[0] + redpos[0]) / 2, (bluepos[1] + redpos[1]) / 2]
        dstpos = [dstnode.pos[1] * px_per_unit + px_per_unit // 2, dstnode.pos[0] * px_per_unit + px_per_unit // 2]
        center_dst_dis = GetDistance(centerpos, dstpos)
        center_to_dst = [dstpos[0] - centerpos[0], dstpos[1] - centerpos[1]]
        rv_center_to_dst = [center_to_dst[1], -center_to_dst[0]]
        blue_to_red = [redpos[0] - bluepos[0], redpos[1] - bluepos[1]]
        dotmul = rv_center_to_dst[0] * blue_to_red[0] + rv_center_to_dst[1] * blue_to_red[1]

        blue_dst_dis = GetDistance(bluepos, dstpos)
        red_dst_dis = GetDistance(redpos, dstpos)
        red_blue_dst_angle = math.acos((math.pow(brdist, 2) + math.pow(blue_dst_dis, 2) - math.pow(red_dst_dis, 2)) / (
                2 * brdist * blue_dst_dis)) * 180 / math.pi
        print('destinatino position: ', dstnode.pos)
        print('angle: ', red_blue_dst_angle)
        print('left or right: ', dotmul)
        if GetDistance(centerpos, dstpos) <= 20:
            if dstnode.pos == finalpos:
                print('task finish')
                break
            elif dstnode._l_son is not None and dstnode._l_son.to_end is True:
                dstnode = dstnode._l_son
            else:
                dstnode = dstnode._r_son
        if red_blue_dst_angle >= 10:
            if dotmul > 0:
                print('too left')
                TurnRight(red_blue_dst_angle)
            else:
                print('too right')
                TurnLeft(red_blue_dst_angle)
        else:
            GoAhead(centerpos)'''

    if pic2.any():
        cv2.imshow('pic2', pic2)
    cv2.imshow('frame', frame)
    cv2.imshow('grayimg', ptfer0)
    c = cv2.waitKey(100) & 0xFF
    if c == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()

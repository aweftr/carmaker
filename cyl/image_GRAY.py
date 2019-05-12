import numpy as np
import cv2

# cap=cv2.VideoCapture(0)
video = "http://admin:admin@192.168.137.212:8081/"
cap = cv2.VideoCapture(video)

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
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit + px_per_unit // 30] == 0:
            self.state[3] = 1  # 状态为黑色是1
        if ptfer[self.pos[0] * px_per_unit + px_per_unit-1, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
            self.state[2] = 1
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit + px_per_unit-1] == 0:
            self.state[1] = 1
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 30, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
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
        cv2.line(frame, pos1, pos2, (255,0,0), 1)

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
            if tmp1._l_son.to_end is True:
                tmp2 = tmp1._l_son
            else:
                tmp2 = tmp1._r_son
            pos1 = (tmp1.pos[1] * px_per_unit + px_per_unit // 2, tmp1.pos[0] * px_per_unit + px_per_unit // 2)
            pos2 = (tmp2.pos[1] * px_per_unit + px_per_unit // 2, tmp2.pos[0] * px_per_unit + px_per_unit // 2)
            cv2.line(frame, pos1, pos2, (255,255,0), 3)
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


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_click_position)

# 主循环
while True:
    ret,frame=cap.read()
    # frame = cv2.imread("figure\\maze1.jpg")
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret1, ptfer = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)

    for i in range(len(position)):
        cv2.circle(frame, (position[i][0], position[i][1]), 5, (255, 0, 0), -1)

    if counter == 4:
        unit_grid = int(input("Please input the grid unit: "))
        px_per_unit = 240 // unit_grid

        iposition = np.float32(position)
        M = cv2.getPerspectiveTransform(iposition, tposition)
        thresh1 = cv2.warpPerspective(gray_img, M, (240, 240))
        ret1, ptfer = cv2.threshold(thresh1, 180, 255, cv2.THRESH_BINARY)
        startpos = [0, 3]
        finalpos = [3, 0]
        a = tree(startpos)
        a.search_Node(3, a.head, finalpos)
        cv2.imshow('ptfer', ptfer)
        counter = 0
        position = []
    '''px_per_unit = 240 // 4
    startpos = [3, 0]
    finalpos = [0, 3]
    a = tree(startpos)
    a.search_Node(3, a.head, finalpos)
    a.findPathToEnd(finalpos)'''

    cv2.imshow('frame', frame)
    # cv2.imshow('what', thresh1)
    # cv2.imshow('ptefer', ptfer)
    c = cv2.waitKey(100) & 0xFF
    if c == ord('q'):
        break
    elif c == ord('c'):
        position = []
        counter = 0

# cap.release()
cv2.destroyAllWindows()

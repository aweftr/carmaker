import numpy as np
import cv2
import math
import serial
import serial.tools.list_ports
import time

# connect to the car via bluetooth.
ports = list(serial.tools.list_ports.comports())
for i in ports:
    print(i[0])
ser = serial.Serial(port=ports[1][0])

# open the external camera
cap = cv2.VideoCapture(1)

# the position of the maze after perspective transform.
tposition = np.float32([[0, 0], [240, 0], [0, 240], [240, 240]])
# the number of pixels on a side in a unit, a unit is a grid in the maze.
px_per_unit = 0


# node defination
class Node():

    def __init__(self, pos, l_son=None, r_son=None):
        self.pos = pos                  # the position of the node
        self.state = [0, 0, 0, 0]       # the state of the four edges around the node.
                                        # 1 if the edge exist and 0 if not.
                                        # -----0-----
                                        # |         |
                                        # |         |
                                        # 3         1
                                        # |         |
                                        # |         |
                                        # -----2-----

        self.figure_state()             # figure the state of the node.
        self._l_son = l_son             # the l_son and r_son contains the adjecent node to which this node can go
        self._r_son = r_son
        self.to_end = False             # True means that through this node the car can goto the final position.
        self.len = 0                    # the number of edges of a node
        for i in range(4):
            if self.state[i] == 1:
                self.len = self.len + 1
        self.rnode = self.Is_a_rnode()  # Is the node an rnode? Rnode means that the node is in the key position
                                        # in which the car may turn its position.

    def figure_state(self):
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit] == 0:
            self.state[3] = 1  # there is an edge in the left hand side of the node.
        if ptfer[self.pos[0] * px_per_unit + px_per_unit - 1, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
            self.state[2] = 1  # there is an edge in the bottom of the node.
        if ptfer[self.pos[0] * px_per_unit + px_per_unit // 2, self.pos[1] * px_per_unit + px_per_unit - 1] == 0:
            self.state[1] = 1  # there is an edge in the right hand side of the node.
        if ptfer[self.pos[0] * px_per_unit, self.pos[1] * px_per_unit + px_per_unit // 2] == 0:
            self.state[0] = 1  # there is an edge in the top of the node.

    def Is_a_rnode(self):
        # Usually, a node which have three edges cannot be an rnode.
        # Because it is a dead end.
        # And a node with one edge can be an rnode, for there are two direction the car can turn to.
        # If a node with two edges and these two edges are adjacent, then this is a rnode.
        # Since it is a corner and the car have one direction to turn to.
        # What's more, if the edges are not adjacent, this is not a rnode. Because the car can only go straight.
        if self.len == 1:
            return True
        if self.len == 2:
            if (self.state[0] == self.state[2] == 1) or (self.state[1] == self.state[3] == 1):
                return False
            else:
                return True
        if self.len == 3:
            return False


# a tree of the nodes.
class tree():
    # the tree has a node head, which is node of the start position of the maze.
    def __init__(self, start_pos):
        self.head = Node(start_pos)

    # A tool function to draw lines between the center of two nodes.
    # This is used to draw the dfs line in the search_Node function below.
    def drawline(self, startpos, finalpos):
        pos1 = (startpos[1] * px_per_unit + px_per_unit // 2, startpos[0] * px_per_unit + px_per_unit // 2)
        pos2 = (finalpos[1] * px_per_unit + px_per_unit // 2, finalpos[0] * px_per_unit + px_per_unit // 2)
        cv2.line(pic1, pos1, pos2, (255, 0, 0), 1)

    # Use dfs to search the rnode in the maze.
    # @initial: the position of in-edge of this node. The car in this node can only turn to
    # @node:    the node which will be searched in the function.
    # @finalpos: final position of the maze. Used to terminate the dfs search.
    def search_Node(self, initial, node, finalpos):

        # search for every edge of this node except the in-edge.
        for i in range(4):
            if i == initial:
                pass    # Skip the in-edge.
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

    # After the dfs search, call the findPathToEnd function to find the path from the head node
    # to the node of the end position. It is actually a warpper function, which call the
    # pfindPathToEnd to get the result.
    #
    # @finalpos: the final position of the maze.
    #
    def findPathToEnd(self, finalpos):
        self.head.to_end = True
        tmp1 = self.head
        self.pfindPathToEnd(self.head, finalpos)
        while True:
            if tmp1._l_son is not None and tmp1._l_son.to_end is True:
                tmp2 = tmp1._l_son
            else:
                tmp2 = tmp1._r_son
            # draw the line which can go out the maze.
            pos1 = (tmp1.pos[1] * px_per_unit + px_per_unit // 2, tmp1.pos[0] * px_per_unit + px_per_unit // 2)
            pos2 = (tmp2.pos[1] * px_per_unit + px_per_unit // 2, tmp2.pos[0] * px_per_unit + px_per_unit // 2)
            cv2.line(pic1, pos1, pos2, (255, 255, 0), 3)
            tmp1 = tmp2
            if tmp2.pos == finalpos:
                break

    # a recursive function which can get the path to the end point of the maze.
    # @inode: search the path from this node to the end point.
    # @finalpos: the final position of the maze.
    def pfindPathToEnd(self, inode, finalpos):
        if inode.pos == finalpos:
            inode.to_end = True     # the node is actually the node of end point.
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


# the HSV range of the value which are used to get the car and the maze automatically.
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
lower_red = np.array([0, 43, 46])
upper_red = np.array([10, 255, 255])
lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])


# Use blue and red to get the position of the car.
# It returns two lists which contains the center position of blue and red.
#
# @frame: the frame of the maze after perspective transform.
#
def GetCarPosition(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    resblue = cv2.inRange(hsv, lower_blue, upper_blue)
    resred = cv2.inRange(hsv, lower_red, upper_red)

    # find the contours of blue and red.
    cntblue = cv2.findContours(resblue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntred = cv2.findContours(resred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    centerblue = centerred = None   # The center position of blue and red contours.

    if len(cntblue) > 0:
        cblue = max(cntblue, key=cv2.contourArea)   # Find the maximum contour of blue.
        ((x, y), radius) = cv2.minEnclosingCircle(cblue)
        M = cv2.moments(cblue)      # Calculate its moments.
        centerblue = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))   # Calculate the center position of blue.
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, centerblue, 5, (0, 255, 0), -1)

    if len(cntred) > 0:
        cred = max(cntred, key=cv2.contourArea) # The same as blue.
        ((x1, y1), radius1) = cv2.minEnclosingCircle(cred)
        M1 = cv2.moments(cred)
        centerred = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
        cv2.circle(frame, (int(x1), int(y1)), int(radius1), (0, 255, 255), 2)
        cv2.circle(frame, centerred, 5, (0, 255, 0), -1)

    cv2.imshow('maskblue', resblue)
    cv2.imshow('maskred', resred)
    return centerblue, centerred


# A tool function to remove an array element which is in a list.
def RmFromListOfArray(origin_list, arrayin):
    for i in range(len(origin_list)):
        if np.array_equal(origin_list[i], arrayin):
            origin_list.pop(i)
            break


# Get the distance from this pixel to the original pixel (0,0).
def disFromOri(ele):
    return ele[0]+ele[1]


# Use green to get the maze.
def GetTheMaze(inframe):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    cnt_green = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    j = 0
    fournode = []       # The center position of the four corners of the maze.

    # Find the largest four contours which represents the four corners of the maze.
    while len(cnt_green) > 0 and j < 4:
        i = max(cnt_green, key=cv2.contourArea)
        j = j + 1
        RmFromListOfArray(cnt_green, i)
        M = cv2.moments(i)
        centeryellow = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        fournode.append(centeryellow)
    fournode.sort(key=disFromOri)       # Sort them by there distance to the original pixel.
    if fournode[1][0] < fournode[2][0]:
        tmp = fournode[1]
        fournode[1] = fournode[2]
        fournode[2] = tmp
    return np.float32(fournode)


# Get the distance between two pixels.
def GetDistance(inpos1, inpos2):
    return math.sqrt(math.pow((inpos1[0] - inpos2[0]), 2) + math.pow((inpos1[1] - inpos2[1]), 2))


# Control the car to turn right
def TurnRight(angle):
    print('TurnRight')
    pip = 'e'
    ser.write(pip.encode())


# Control the car to turn left
def TurnLeft(angle):
    print('TurnLeft')
    pip = 'd'
    ser.write(pip.encode())


# Control the car to go ahead.
def GoAhead(distance):
    print('GoAhead')
    pip = 'f'
    ser.write(pip.encode())


cv2.namedWindow('grayimg')
cv2.createTrackbar('mask', 'grayimg', 106, 255, nothing)

pic2 = np.array([0])

# The maze is a 4 * 4 maze.
unit_grid = 4

# the number of pixels on a side in a unit, a unit is a grid in the maze.
px_per_unit = 240 // unit_grid

c1 = 'n'
# start and end position of the maze.
#   -----------------
#   |0,0|0,1|0,2|0.3|
#   -----------------
#   |1,0|1,1|1,2|1.3|
#   -----------------
#   |2,0|2,1|2,2|2.3|
#   -----------------
#   |3,0|3,1|3,2|3.3|
#   -----------------
startpos = [3, 0]
finalpos = [0, 3]

# 主循环
while True:
    ret, frame = cap.read()     # Read the external camera.
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change the frame to a gray-scale frame.
    mask_value = cv2.getTrackbarPos('mask', 'grayimg')
    ret0, ptfer0 = cv2.threshold(gray_img, mask_value, 255, cv2.THRESH_BINARY)  # Binarization the gray_img.
    iposition = GetTheMaze(frame)   # get the position of the maze.

    M = cv2.getPerspectiveTransform(iposition, tposition)   # Get the transform matrix used to transform the maze.
    pic1 = cv2.warpPerspective(frame, M, (240, 240))    # Change the maze to a 240*240 square.
    pic2 = pic1.copy()

    thresh1 = cv2.warpPerspective(gray_img, M, (240, 240))  # Change the maze in the gray_img to a 240*240 square.
    ret1, ptfer = cv2.threshold(thresh1, mask_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(0, (5, 5))   # Do some erode to eliminate the little white dot
    ptfer = cv2.erode(ptfer, kernel, iterations=2)  # and strengthen the edges of a node.

    # Press 'y' to get the position of the maze which is correct.
    while(c1 != ord('y') and c1 != ord('k')):
        ret, frame = cap.read()
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

        # Use the search_Node in the tree data structure to search the maze.
        a.search_Node(3, a.head, finalpos)

        # find the right path to the end position.
        a.findPathToEnd(finalpos)

        # Set up the first rnode the car should goto.
        if a.head._l_son is not None and a.head._l_son.to_end is True:
            dstnode = a.head._l_son
        else:
            dstnode = a.head._r_son

        cv2.imshow('ptefer', ptfer)
        cv2.imshow('pic1', pic1)
        c1 = ord('k')
    elif c1 == ord('n'):
        print('Please select maze again')
        cv2.destroyWindow('ptefer')
        pic2 = np.array([0])

    if pic2.any():
        pic2 = cv2.warpPerspective(frame, M, (240, 240))

        # Use GetCarPosition to get the center positions of blue and red in the car.
        bluepos, redpos = GetCarPosition(pic2)

        # Get the distance between the center positions of blue and red in the car.
        brdist = GetDistance(bluepos, redpos)

        # Center of the center positions of blue and red, which is also the center position of the car.
        centerpos = [(bluepos[0] + redpos[0]) / 2, (bluepos[1] + redpos[1]) / 2]

        # The position of the current destination which the car should go to.
        dstpos = [dstnode.pos[1] * px_per_unit + px_per_unit // 2, dstnode.pos[0] * px_per_unit + px_per_unit // 2]

        # The distance between the center of the car and the current destination.
        center_dst_dis = GetDistance(centerpos, dstpos)

        # The vector from center of the car to destinatino.
        center_to_dst = [dstpos[0] - centerpos[0], dstpos[1] - centerpos[1]]

        # The center_to_dst vector turned -90°, which is used to estimate whether the car should turn left or right.
        rv_center_to_dst = [center_to_dst[1], -center_to_dst[0]]

        # The vector from the center of blue to red, which represent the direction of the car.
        blue_to_red = [redpos[0] - bluepos[0], redpos[1] - bluepos[1]]

        # dot product of the vector rv_center_to_dst and blue_to_red.
        # If it is positive, then the car should turn right.
        # If it is negative, then the car should turn left.
        dotmul = rv_center_to_dst[0] * blue_to_red[0] + rv_center_to_dst[1] * blue_to_red[1]

        blue_dst_dis = GetDistance(bluepos, dstpos)
        red_dst_dis = GetDistance(redpos, dstpos)
        red_blue_dst_angle = math.acos((math.pow(brdist, 2) + math.pow(blue_dst_dis, 2) - math.pow(red_dst_dis, 2)) / (
                2 * brdist * blue_dst_dis)) * 180 / math.pi
        print('destinatino position: ', dstnode.pos)
        print('angle: ', red_blue_dst_angle)
        print('left or right: ', dotmul)
        if GetDistance(centerpos, dstpos) <= 20:
            # Which means that the car is close to the current destination.
            # Thus the destinatino should be changed.
            if dstnode.pos == finalpos:
                print('task finish')
                break
            elif dstnode._l_son is not None and dstnode._l_son.to_end is True:
                dstnode = dstnode._l_son
            else:
                dstnode = dstnode._r_son

        # If the angle between the direction of the car and the direction from blue to destination
        # is too large, the car should decrease the angle to 10 and then it can go ahead.
        if red_blue_dst_angle >= 10:
            if dotmul > 0:
                print('too left')
                TurnRight(red_blue_dst_angle)
            else:
                print('too right')
                TurnLeft(red_blue_dst_angle)
        else:
            GoAhead(centerpos)

    if pic2.any():
        cv2.imshow('pic2', pic2)
    cv2.imshow('frame', frame)
    cv2.imshow('grayimg', ptfer0)
    c = cv2.waitKey(100) & 0xFF
    if c == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

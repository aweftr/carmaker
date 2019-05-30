import cv2
import numpy as np

lower_yellow = np.array([35, 43, 46])
upper_yellow = np.array([77, 255, 255])
# cap = cv2.VideoCapture(0)
center = []

def RmFromListOfArray(origin_list, arrayin):
    for i in range(len(origin_list)):
        if np.array_equal(origin_list[i], arrayin):
            origin_list.pop(i)
            break

def disFromOri(ele):
    return ele[0]+ele[1]


frame = cv2.imread("maze_4_node.png")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
cnt_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
j = 0

while len(cnt_yellow) > 0 and j <4:
    i = max(cnt_yellow, key=cv2.contourArea)
    j = j + 1
    RmFromListOfArray(cnt_yellow, i)
    M = cv2.moments(i)
    centerblue = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
    center.append(centerblue)
center.sort(key=disFromOri)
if center[1][0] > center[2][0]:
    tmp = center[1]
    center[1] = center[2]
    center[2] = tmp

cv2.imshow('frame', frame)
c = cv2.waitKey(0) & 0xFF
if c == 'q':
    cv2.destroyWindow(frame)

cv2.destroyAllWindows()

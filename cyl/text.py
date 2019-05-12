import cv2
cv2.namedWindow("camera", 1)
video = "http://admin:admin@192.168.137.212:8081/"
cap = cv2.VideoCapture(video)
while True:
    ret,what = cap.read()
    cv2.imshow('gaga', what)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()

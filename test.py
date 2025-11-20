import cv2
cap = cv2.VideoCapture(0)
while True:
    key,img = cap.read()
    cv2.imshow("video",img)
    cv2.waitKey(1)
    if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
        break

    
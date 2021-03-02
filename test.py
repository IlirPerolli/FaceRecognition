import cv2

img = cv2.imread("s1.jpg",0)

while True:


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image', img)
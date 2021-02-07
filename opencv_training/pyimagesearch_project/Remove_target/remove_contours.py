import numpy as np
import imutils
import cv2

def is_contour_bad(c):

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, peri*0.02, True)

    return not len(approx) == 4


image = cv2.imread("shapes.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("Original", image)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cv2.CHAIN_APPROX_SIMPLE only leave end points
cnts = imutils.grab_contours(cnts)
mask = np.ones(image.shape[:2], dtype="uint8") * 255

for c in cnts:
    if is_contour_bad(c):
        cv2.drawContours(mask, [c], -1, 0, -1)

image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("After", image)
cv2.waitKey(0)
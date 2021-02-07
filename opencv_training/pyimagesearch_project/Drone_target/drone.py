import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "Path to the video file")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])

while True:

    (grabbed, frame) = camera.read() # read return two value, grabbed is boolean, frame is mat
    status = "No Targets"

    if not grabbed:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # detect cv2 or cv4

    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        if len(approx) >= 4 and len(approx) <= 6:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)
            area = cv2.contourArea(c)
            hullarea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullarea)

            keepDims = w > 25 and h > 25
            keepSolity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

            if keepDims and keepSolity and keepAspectRatio:

                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                status = "Target(s) Acquired"

                M = cv2.moments(approx)
                # get centroid
                (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
                (startX, endX) = (int(cX - (w*0.15)), int(cX + (w*0.15)))
                (startY, endY) = (int(cY - (h*0.15)), int(cY + (h*0.15)))
                cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
                cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


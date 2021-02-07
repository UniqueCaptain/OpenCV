from facedetector import FaceDetector
import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "Path to the face cascade file")
ap.add_argument("-v", "--video", help = "Path to the video file")
args = vars(ap.parse_args())

fd = FaceDetector(args["face"])

if not args.get("video", False):
    camera = cv2.VideoCapture(-1)

else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read() # grabbed is boolean

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
    frameClone = frame.copy()

    for (fx, fy, fw, fh) in faceRects:
        cv2.rectangle(frameClone, (fx, fy), ((fx + fw), (fy + fh)), (0, 255, 0), 2)

    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


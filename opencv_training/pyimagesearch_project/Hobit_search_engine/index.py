from rgbhistogram import RGBHistogram
from imutils.paths import list_images
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to the directory that contains the image to be indexed")
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

index = {}
desc = RGBHistogram([8, 8, 8])
for imagePath in list_images(args["dataset"]):
    k = imagePath[imagePath.rfind("/") + 1:]

    image = cv2.imread(imagePath)
    features = desc.describe(image)
    index[k] = features
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

print("[INFO] done...index {} images".format(len(index)))

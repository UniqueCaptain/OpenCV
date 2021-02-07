from pyimagesearch.searcher import Searcher
import numpy as np
import argparse
import os
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
    help = "Path to where we stored our index")
args = vars(ap.parse_args())

index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)
for (query, queryFeatures) in index.items():
    results = searcher.search(queryFeatures)
    path = os.path.join(args["dataset"], query)
    queryImage = cv2.imread(path)
    cv2.imshow("Query", queryImage)
    print("query: {}".format(query))

    montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
    montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")

    for j in range(0, 10):
        (score, imageName) = results[j]
        path = os.path.join(args["dataset"], imageName)
        result = cv2.imread(path)
        print("¥t{}. {} : {:.3f}".format(j + 1, imageName, score))

        if j < 5:
            montageA[j * 166:(j + 1) * 166, :] = result
        else:
            montageB[(j - 5) * 166: ((j - 5) + 1) * 166, :] = result
    cv2.imshow("Result 1-5", montageA)
    cv2.imshow("Result 6-10", montageB)
    cv2.waitKey(0)

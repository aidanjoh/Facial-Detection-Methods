import sys
import cv2
import dlib
import skimage
import csv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

"""
Takes in an image path and coverts the image to grayscale.

"""
def convertImageToGrayScale(imPath=None):
    if imPath is None:
        print("No image path was passed!")
        return
    
    image = cv2.imread(imPath)
    grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayScaleImage

"""
Plots an image in grayscale.

"""
def viewImage(im=None):
    if im is None:
        print("No image was passed!")
        return

    #image = plt.imread(im)
    plt.imshow(im,cmap='gray', vmin=0, vmax=255)
    plt.show()
    return

"""
Performs the Haars cascade facial detection. Different parameters can be passed in to 
change the performance of the haars facial detection algorithm. Returns all the different
faces detected.
"""
def haarsFaceDetect(faceCascade, grayImage, minSize, scaleFactor=1.1, minNeighbors=5):
    faces = faceCascade.detectMultiScale(
                                            grayImage,
                                            scaleFactor=scaleFactor,
                                            minNeighbors=minNeighbors,
                                            minSize=minSize,
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                        )
    return faces

"""
Performs the LBP-based Haars cascade for frontal facial detection using OpenCV. Different parameters can be passed in to change the performance of the haars facial detection algorithm. Returns all the different faces detected. 
"""
def lbpHaarsFaceDetectCV(grayImage, minSize, scaleFactor=1.1, minNeighbors=5):
    # Load LBP cascade classifier training
    lbpFaceCascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    faces = lbpFaceCascade.detectMultiScale(
                                            grayImage,
                                            scaleFactor=scaleFactor,
                                            minNeighbors=minNeighbors,
                                            minSize=minSize,
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                        )
    return faces

"""
Performs the LBP-based Haars cascade for frontal facial detection using Skimage. Different parameters can be passed in to change the performance of the haars facial detection algorithm. Returns all the different faces detected. 

Output Dicts have form {'r': int, 'c': int, 'width': int, 'height': int}, where 'r' represents row position of top left corner of detected window, 'c' - col position, 'width' - width of detected window, 'height' - height of detected window.
"""
def lbpHaarsFaceDetectSki(grayImage, min_size, max_size, scale_factor=1.1, step_ratio=1.1 ):
    # Load LBP cascade classifier training
    lbpFaceCascade = skimage.feature.Cascade('lbpcascade_frontalface_improved.xml')
    faces = lbpFaceCascade.detect_multi_scale(
                                            grayImage,
                                            scale_factor=scale_factor,
                                            step_ratio=step_ratio,
                                            min_size=min_size,
                                            max_size=max_size,
                                        )
    return faces 


"""
Calculates the interzection over union between the ground truth box of an image and the
predicted box for face detection.

"""
def calcIntersectiontionOverUnion(groundTruthBox, predictedBox):
    print(groundTruthBox)
    print(predictedBox)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(groundTruthBox[0], predictedBox[0])
    yA = max(groundTruthBox[1], predictedBox[1])
    xB = min(groundTruthBox[2], predictedBox[2])
    yB = min(groundTruthBox[3], predictedBox[3])
    print(f"xA = {xA}, yA = {yA}, xB = {xB}, yB = {yB}")

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA+1) * max(0, yB - yA+1)
    print(f"The inter Area is {interArea}")

    # compute the area of both the prediction and ground-truth
    # rectangles
    groundTruthBoxArea = (groundTruthBox[2] - groundTruthBox[0] + 1) * (groundTruthBox[3] - groundTruthBox[1] + 1)
    print(f"Ground truth area is {groundTruthBoxArea}")
    predictedBoxArea = (predictedBox[2] - predictedBox[0] + 1) * (predictedBox[3] - predictedBox[1] + 1)
    print(f"Predicted area is {predictedBoxArea}")
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(groundTruthBoxArea + predictedBoxArea - interArea)
    # return the intersection over union value
    return iou

def main():
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    groundTrueBoundingBox = [[] for i in range(10)]    #Change this to be based on the number of photos (a variable of length of photos)

    with open("updatedGroundTruthBBox.csv") as csvFile:
        groundTruthBoundingBoxes = csv.reader(csvFile, delimiter=',')

        rowCount = 0
        for groundTruthBoundingBox in groundTruthBoundingBoxes:
            if rowCount == 0:
                rowCount += 1
                continue

            imageNumber = int(groundTruthBoundingBox[0])
            coordinates = groundTruthBoundingBox[1:]
            intCoordinates = [int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])]

            groundTrueBoundingBox[imageNumber].append(intCoordinates)
            rowCount += 1


    for imageNum in range(1,10):
        imagePath = f"Images/personal/{imageNum}.jpg"
        grayImage = convertImageToGrayScale(imagePath)
        faces = lbpHaarsFaceDetectSki(grayImage, min_size=(30,30), max_size=(1000,1000), scale_factor=1.1, step_ratio=1.4)
        # print(faces[0]['r'])
        print(faces)
         
        plt.imshow(grayImage)
        img_desc = plt.gca()
        plt.set_cmap('gray')
        
       

        for patch in faces:

            if len(groundTrueBoundingBox[imageNum]) == 1:
                predictedBox = [patch['c'], patch['r'], patch['c']+ patch['width'], patch['r']+ patch['height']]
                print(groundTrueBoundingBox[imageNum][0])
                cv2.rectangle(grayImage, (groundTrueBoundingBox[imageNum][0][0], groundTrueBoundingBox[imageNum][0][1]), \
                                        (groundTrueBoundingBox[imageNum][0][2], groundTrueBoundingBox[imageNum][0][3]), (0, 0, 255), 5)
                iouScore = calcIntersectiontionOverUnion(np.array(groundTrueBoundingBox[imageNum][0]), predictedBox)
                print(f"The IoU Score is: {iouScore}")
            else:
                pass
            # plotting boundaries on image
            
            img_desc.add_patch(
                patches.Rectangle(
                    (patch['c'], patch['r']),
                    patch['width'],
                    patch['height'],
                    fill=False,
                    color='r',
                    linewidth=2
                )
            )

        plt.show()
    

    for imageNum in range(1,10):
        imagePath = f"Images/personal/{imageNum}.jpg"
        grayImage = convertImageToGrayScale(imagePath)
        #viewImage(grayImage)
        faces = haarsFaceDetect(faceCascade, grayImage, minSize=(30,30), scaleFactor=1.1, minNeighbors=5)
        
        print(faces)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(grayImage, (x, y), (x+w, y+h), (0, 255, 0), 5)
            predictedBox = [x, y, x+w, y+h]
            if len(groundTrueBoundingBox[imageNum]) == 1:
                print(groundTrueBoundingBox[imageNum][0])
                cv2.rectangle(grayImage, (groundTrueBoundingBox[imageNum][0][0], groundTrueBoundingBox[imageNum][0][1]), \
                                        (groundTrueBoundingBox[imageNum][0][2], groundTrueBoundingBox[imageNum][0][3]), (0, 0, 255), 5)
                iouScore = calcIntersectiontionOverUnion(np.array(groundTrueBoundingBox[imageNum][0]), predictedBox)
                print(f"The IoU Score is: {iouScore}")
            else:
                pass

        # Display the resulting picture with the detected bounding box(es)
        cv2.imshow('Face Detection', grayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    for imageNum in range(1,10):
        imagePath = f"Images/personal/{imageNum}.jpg"

        img = dlib.load_rgb_image(imagePath)
        detectedFaces = detector(img, 1)
        print("Number of faces detected: {}".format(len(detectedFaces)))

        for i, d in enumerate(detectedFaces):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            predictedBox = [d.left(), d.top(), d.right(), d.bottom()]
            if len(groundTrueBoundingBox[imageNum]) == 1:
                print(groundTrueBoundingBox[imageNum][0])
                iouScore = calcIntersectiontionOverUnion(np.array(groundTrueBoundingBox[imageNum][0]), predictedBox)
                print(f"The IoU Score is: {iouScore}")
            else:
                pass
        
        plt.imshow(img)
        img_desc = plt.gca()
        plt.set_cmap('gray')

        for patch in faces:

            img_desc.add_patch(
                patches.Rectangle(
                    (patch['c'], patch['r']),
                    patch['width'],
                    patch['height'],
                    fill=False,
                    color='r',
                    linewidth=2
                )
            )

        plt.show()

        # Draw a rectangle around the faces
        win.clear_overlay()
        win.set_image(img)
        print(detectedFaces)
        win.add_overlay(detectedFaces)
        win.add_overlay([groundTrueBoundingBox[imageNum][0][0], groundTrueBoundingBox[imageNum][0][1],
                        groundTrueBoundingBox[imageNum][0][2], groundTrueBoundingBox[imageNum][0][3]])
        dlib.hit_enter_to_continue()

        # Finally, if you really want to you can ask the detector to tell you the score
        # for each detection.  The score is bigger for more confident detections.
        # The third argument to run is an optional adjustment to the detection threshold,
        # where a negative value will return more detections and a positive value fewer.
        # Also, the idx tells you which of the face sub-detectors matched.  This can be
        # used to broadly identify faces in different orientations.

        dets, scores, idx = detector.run(img, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))

        for (x, y, w, h) in faces:
            cv2.rectangle(grayImage, (x, y), (x+w, y+h), (0, 255, 0), 5)



if __name__ == "__main__":
    main()
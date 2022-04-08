import sys
import cv2
import dlib
import skimage

import matplotlib.pyplot as plt


def convertImageToGrayScale(imPath=None):
    if imPath is None:
        print("No image path was passed!")
        return
    
    image = cv2.imread(imPath)
    grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayScaleImage


def viewImage(im=None):
    if im is None:
        print("No image was passed!")
        return

    #image = plt.imread(im)
    plt.imshow(im,cmap='gray', vmin=0, vmax=255)
    plt.show()
    return

def haarsFaceDetect(faceCascade, grayImage, minSize, scaleFactor=1.1, minNeighbors=5):
    faces = faceCascade.detectMultiScale(
                                            grayImage,
                                            scaleFactor=scaleFactor,
                                            minNeighbors=minNeighbors,
                                            minSize=minSize,
                                            flags=cv2.CASCADE_SCALE_IMAGE
                                        )
    return faces


def main():
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # for imageNum in range(1,11):
    #     imagePath = f"Images/personal/{imageNum}.jpg"
    #     grayImage = convertImageToGrayScale(imagePath)
    #     viewImage(grayImage)
    #     faces = haarsFaceDetect(faceCascade, grayImage, minSize=(30,30), scaleFactor=1.1, minNeighbors=5)

    #     # Draw a rectangle around the faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(grayImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #     # Display the resulting picture with the detected bounding box(es)
    #     cv2.imshow('Face Detection', grayImage)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    f = "Images/personal/1.jpg"
    img = dlib.load_rgb_image(f)
    detectedFaces = detector(img, 1)
    print("Number of faces detected: {}".format(len(detectedFaces)))
    for i, d in enumerate(detectedFaces):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left (), d.top(), d.right (), d.bottom ()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(detectedFaces)
    dlib.hit_enter_to_continue()

    # Finally, if you really want to you can ask the detector to tell you the score
    # for each detection.  The score is bigger for more confident detections.
    # The third argument to run is an optional adjustment to the detection threshold,
    # where a negative value will return more detections and a positive value fewer.
    # Also, the idx tells you which of the face sub-detectors matched.  This can be
    # used to broadly identify faces in different orientations.
    if (len(f) > 0):
        img = dlib.load_rgb_image(f)
        dets, scores, idx = detector.run(img, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))


    imagePath = "Images/personal/10.jpg"
    grayImage = convertImageToGrayScale(imagePath)
    viewImage(grayImage)

    # Get all the faces from the image
    faces = haarsFaceDetect(faceCascade, grayImage, minSize=(30,30), scaleFactor=1.1, minNeighbors=5)
   
    # faces = faceCascade.detectMultiScale(
    #                                         grayImage,
    #                                         scaleFactor=1.1,
    #                                         minNeighbors=5,
    #                                         minSize=(30, 30),
    #                                         flags=cv2.CASCADE_SCALE_IMAGE
    #                                     )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(grayImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting picture with the detected bounding box(es)
    cv2.imshow('Face Detection', grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # with open("Images/YaleDB/yaleB11_P01A+000E+00.pgm", 'rb') as pgmf:
    #     im = plt.imread(pgmf)
    #     plt.imshow(im,cmap='gray', vmin=0, vmax=255)
    #     plt.show()

    # with open("Images/YaleDB/yaleB11_P03A+000E+20.pgm", 'rb') as pgmf:
    #     im = plt.imread(pgmf)
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.show()

    # cascPath = "haarcascade_frontalface_default.xml"
    # newCascPath = "lbpcascade_frontalface_improved.xml"
    # faceCascade = cv2.CascadeClassifier(newCascPath)

    # video_capture = cv2.VideoCapture(0)

    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = video_capture.read()

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     faces = faceCascade.detectMultiScale(
    #         gray,
    #         scaleFactor=1.1,
    #         minNeighbors=5,
    #         minSize=(30, 30),
    #         flags=cv2.CASCADE_SCALE_IMAGE
    #     )

    #     # Draw a rectangle around the faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #     # Display the resulting frame
    #     cv2.imshow('Video', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything is done, release the capture
    # video_capture.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
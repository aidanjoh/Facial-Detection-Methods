import sys
import cv2
import dlib
import skimage

import matplotlib.pyplot as plt



def main():
    with open("Images/YaleDB/yaleB11_P00A+000E+00.pgm", 'rb') as pgmf:
        im = plt.imread(pgmf)
        plt.imshow(im,cmap='gray', vmin=0, vmax=255)
        plt.show()

    with open("Images/YaleDB/yaleB11_P00A+000E+20.pgm", 'rb') as pgmf:
        im = plt.imread(pgmf)
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.show()

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
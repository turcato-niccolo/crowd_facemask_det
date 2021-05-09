from main import *
import numpy as np
import cv2

image = cv2.imread("image.png")
cv2.imshow("Resized gray", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

color_class = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}  # set label (0-incorrect, 1-with, 2-without)
name_class = {0: "incorrect", 1: "with_mask", 2: "without_mask"}

facemask_rec_model = FacemaskRecognitionModel("models/better_facemask_model.h5")
face_cropper_net = FaceCropperResNetSSD()

image_copy = cv2.copyTo(image, None)

faces = face_cropper_net.crop(image)

for face in faces:
    (ROI, confidence) = face
    (x, y, x1, y1) = ROI
    blob = image[y:y1, x:x1]
    if blob.size != 0:
        frame_label = facemask_rec_model.predict_one(blob)
        put_label(image, ROI, name_class[frame_label], color_class[frame_label])

cv.imshow("Face detection", image)

cv.waitKey(0)

import argparse
import cv2 as cv
import time
import numpy as np
from facemask_recognition_model import *

# RESNETSSD_FACEDETECTOR  face detector based on SSD framework with reduced ResNet-10 backbone
# https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/face_detector/how_to_train_face_detector.txt
# https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
# https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt

proto_txt_file_path = 'models/DNN_face_rec/deploy.prototxt.txt'
model_file_path = 'models/DNN_face_rec/res10_300x300_ssd_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(proto_txt_file_path, model_file_path)

parser = argparse.ArgumentParser(description='Code for Face detection.')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

font = cv.FONT_HERSHEY_SIMPLEX
color_class = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}  # set label (0-incorrect, 1-with, 2-without)
name_class = {0: "incorrect", 1: "with_mask", 2: "without_mask"}

prev_frame_time = 0
new_frame_time = 0

facemask_rec_model = facemask_recognition_model("models/facemask_model.h5")

while True:
    ret, frame = cap.read()

    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    faces = []
    (height, width) = frame.shape[:2]

    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype(int)
            h = y1 - y
            w = x1 - x
            x = int(max(x - w / 2, 0))
            y = int(max(y - h / 2, 0))
            x1 = int(min(x1 + w / 2, width - 1))
            y1 = int(min(y1 + h / 2, height - 1))
            # frame_label = facemask_rec_model.predict_one(frame[b_y:b_y1, b_x:b_x1])
            frame_label = facemask_rec_model.predict_one(frame[y:y1, x:x1])
            cv.rectangle(frame, (x, y), (x1, y1), color_class[frame_label], thickness=4)
            cv2.rectangle(frame, (x, y - 40), (x1, y), color_class[frame_label], -1)
            cv2.putText(frame, name_class[frame_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    fps = str(fps)
    cv.putText(frame, fps, (7, 30), font, 1, (100, 255, 0), 3, cv.LINE_AA)

    cv.imshow("Face detection", frame)

    if cv.waitKey(1) == 27:
        break

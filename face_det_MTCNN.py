from __future__ import print_function

import time
import cv2 as cv
import argparse
import tensorflow as tf
from mtcnn import MTCNN


def detect(_frame, _detector):
    frame_rgb = cv.cvtColor(_frame, cv.COLOR_BGR2RGB)
    boxes = _detector.detect_faces(frame_rgb)
    _faces = []
    for box in boxes:
        b = box['box']
        _faces.append([b[0], b[1], b[2], b[3]])
    return _faces


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

with tf.device('GPU:0'):
    detector = MTCNN()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        # frame = cv.resize(frame, (400, 300))

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        faces = detect(frame, detector)
        new_frame_time = time.time()
        for (x, y, w, h) in faces:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=4)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        fps = str(fps)
        cv.putText(frame, fps, (7, 30), font, 1, (100, 255, 0), 3, cv.LINE_AA)

        cv.imshow("Face detection", frame)

        if cv.waitKey(1) == 27:
            break

import argparse
import cv2 as cv
import time
import numpy as np

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

prev_frame_time = 0
new_frame_time = 0

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
            frame = cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), thickness=4)

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

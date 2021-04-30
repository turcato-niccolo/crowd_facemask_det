import argparse
import cv2 as cv
import time
import numpy as np
from face_recognition_model import *
from face_cropper import *


def bind_camera(parser):
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()

    camera_device = args.camera
    # -- 2. Read the video stream
    cap = cv.VideoCapture(camera_device)

    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    return cap


def update_fps(frame, prev_frame_time, font=cv.FONT_HERSHEY_SIMPLEX):
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    cv.putText(frame, str(fps), (7, 30), font, 1, (100, 255, 0), 3, cv.LINE_AA)
    return new_frame_time


def put_label(frame, ROI, label, color, font=cv.FONT_HERSHEY_SIMPLEX):
    (x, y, x1, y1) = ROI
    cv2.rectangle(frame, (x, y), (x1, y1), color, thickness=4)
    cv2.rectangle(frame, (x, y - 40), (x1, y), color, -1)
    cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 255, 255), 2)


def start(camera):
    color_class = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}  # set label (0-incorrect, 1-with, 2-without)
    name_class = {0: "incorrect", 1: "with_mask", 2: "without_mask"}

    facemask_rec_model = FacemaskRecognitionModel("models/facemask_model.h5")
    face_cropper_net = FaceCropperResNetSSD()

    frame_time = time.time()
    while True:
        ret, frame = camera.read()
        (height, width) = frame.shape[:2]
        faces = face_cropper_net.crop(frame)

        for face in faces:
            (ROI, confidence) = face
            (x, y, x1, y1) = ROI
            blob = frame[y:y1, x:x1]
            if blob.size != 0:
                frame_label = facemask_rec_model.predict_one(blob)
                put_label(frame, ROI, name_class[frame_label], color_class[frame_label])

        frame_time = update_fps(frame, frame_time)
        cv.imshow("Face detection", frame)

        if cv.waitKey(1) == 27:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Face detection.')
    cap = bind_camera(parser)
    start(cap)


class CameraBindFailedError(RuntimeError):
    """
    Exception raised when the binding process of the camera fails.
    """
    def __init__(self):
        super().__init__("the binding process of the camera failed")

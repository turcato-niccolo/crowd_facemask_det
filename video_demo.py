import numpy as np
import cv2
from main import *


def start_video(video_cap, out_write):
    color_class = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}  # set label (0-incorrect, 1-with, 2-without)
    name_class = {0: "incorrect", 1: "with_mask", 2: "without_mask"}

    facemask_rec_model = FacemaskRecognitionModel("models/facemask_model.h5")
    face_cropper_net = FaceCropperResNetSSD()

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        faces = face_cropper_net.crop(frame)

        for face in faces:
            (ROI, confidence) = face
            (x, y, x1, y1) = ROI
            blob = frame[y:y1, x:x1]
            if blob.size != 0:
                frame_label = facemask_rec_model.predict_one(blob)
                put_label(frame, ROI, name_class[frame_label], color_class[frame_label])

        cv.imshow("Face detection", frame)

        out_write.write(frame)

        if cv.waitKey(1) == 27:
            break


cap = cv2.VideoCapture('videos/test_video.mp4')
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (videoWidth, videoHeight))

start_video(cap, out)

cap.release()
out.release()
cv2.destroyAllWindows()

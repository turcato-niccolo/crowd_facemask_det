import numpy as np
import cv2

# RESNETSSD_FACEDETECTOR  face detector based on SSD framework with reduced ResNet-10 backbone
# https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/face_detector/how_to_train_face_detector.txt
# https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
# https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt

class FaceCropperResNetSSD:
    def __init__(self):
        proto_txt_file_path = 'models/DNN_face_rec/deploy.prototxt.txt'
        model_file_path = 'models/DNN_face_rec/res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(proto_txt_file_path, model_file_path)

    def crop(self, image, confidence_threshold=0.5):
        faces = list()
        (height, width) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300))

        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype(int)
                h = y1 - y
                w = x1 - x
                x = int(max(x - w / 3, 0))
                y = int(max(y - h / 3, 0))
                x1 = int(min(x1 + w / 3, width - 1))
                y1 = int(min(y1 + h / 3, height - 1))
                faces.append(((x, y, x1, y1), confidence))  # TODO may exist something like a ROI o a Rectangle
        return faces

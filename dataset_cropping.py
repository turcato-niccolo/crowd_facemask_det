import cv2
from facemask_recognition_model import *
import numpy as np
import glob
import os

start_id = 3000
in_paths = ["C:/Users/roberto/Downloads/www.pexels.com/glasses"]
out_path = "C:/Users/roberto/Downloads/www.pexels.com/cropped_dataset"

np.random.seed(42)
folder_class = ["/incorrect_mask", "/with_mask", "/without_mask"]
out_img_ext = ".jpg"
mask_recognition_model_path = "models/facemask_model.h5"
proto_txt_file_path = 'models/DNN_face_rec/deploy.prototxt.txt'
model_file_path = 'models/DNN_face_rec/res10_300x300_ssd_iter_140000.caffemodel'
saved_images = 0

# make the directories in output folder id they don't exist
for folder in folder_class:
    if not os.path.exists(out_path + folder):
        os.makedirs(out_path + folder)

# load nets
net = cv2.dnn.readNetFromCaffe(proto_txt_file_path, model_file_path)
facemask_rec_model = facemask_recognition_model(mask_recognition_model_path)

for folder in in_paths:
    # load images
    images_bgr = [cv2.imread(file) for file in glob.glob(folder + "/*")]
    images_bgr = [x for x in images_bgr if x is not None]
    num_images = len(images_bgr)

    for j in range(num_images):
        print(str(j) + " / " + str(num_images))

        curr_img = images_bgr[j]
        (height, width) = curr_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(curr_img, (300, 300)), 1.0, (300, 300))

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

                if x < width and y < height and x1 < width and y1 < height:
                    img = curr_img[y:y1, x:x1]  # [row, column]
                    if img.shape[0] > 0 and img.shape[1] > 0:
                        frame_label = facemask_rec_model.predict_one(img)
                        name = "/" + str(start_id + saved_images) + out_img_ext
                        cv2.imwrite(out_path + folder_class[frame_label] + name, img)
                        saved_images = saved_images + 1

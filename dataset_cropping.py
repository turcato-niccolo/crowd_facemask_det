import cv2
from face_recognition_model import *
from face_cropper import *
import numpy as np
import glob
import os

start_id = 3000
in_paths = ["DATA"]
out_path = "cropped_dataset"

np.random.seed(42)
folder_class = ["/incorrect_mask", "/with_mask", "/without_mask"]
out_img_ext = ".jpg"
mask_recognition_model_path = "models/facemask_model.h5"
saved_images = 0

# make the directories in output folder id they don't exist
for folder in folder_class:
    if not os.path.exists(out_path + folder):
        os.makedirs(out_path + folder)

# load nets
facemask_rec_model = FacemaskRecognitionModel(mask_recognition_model_path)
face_cropper_net = FaceCropperResNetSSD()
cnt = 1
for folder in in_paths:
    # load images
    images_bgr = [cv2.imread(file) for file in glob.glob(folder + "/*")]
    images_bgr = [x for x in images_bgr if x is not None]
    num_images = len(images_bgr)

    for curr_img in images_bgr:

        (height, width) = curr_img.shape[:2]
        faces = face_cropper_net.crop(curr_img)

        for face in faces:
            ((x, y, x1, y1), confidence) = face
            if x < width and y < height and x1 < width and y1 < height:
                img = curr_img[y:y1, x:x1]  # [row, column]
                if img.shape[0] > 0 and img.shape[1] > 0:
                    frame_label = facemask_rec_model.predict_one(img)
                    name = "/" + str(start_id + saved_images) + out_img_ext
                    cv2.imwrite(out_path + folder_class[frame_label] + name, img)
                    saved_images = saved_images + 1

        print(str(cnt) + " / " + str(num_images))
        cnt = cnt + 1
    cnt = 1

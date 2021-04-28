import cv2
import time
import numpy as np
import glob

np.random.seed(42)

# load images
images_without_bgr = [cv2.imread(file) for file in glob.glob("DATA/senza/*")]
numNo = len(images_without_bgr)
images_correct_bgr = [cv2.imread(file) for file in glob.glob("DATA/con-bene/*")]
numWith = len(images_correct_bgr)
images_incorrect_bgr = [cv2.imread(file) for file in glob.glob("DATA/con-male/*")]
numIncorrect = len(images_incorrect_bgr)

out_path = "cropped_dataset"

images_bgr = images_incorrect_bgr + images_correct_bgr + images_without_bgr
numImages = len(images_bgr)

proto_txt_file_path = 'models/DNN_face_rec/deploy.prototxt.txt'
model_file_path = 'models/DNN_face_rec/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto_txt_file_path, model_file_path)

for j in range(len(images_bgr)):
    print(str(j) + " / " + str(len(images_bgr)))

    curr_img = images_bgr[j]
    (height, width) = curr_img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(curr_img, (300, 300)), 1.0, (300, 300))

    if j in range(numIncorrect):
        class_out_path = out_path + "/mask_weared_incorrect"
    elif j in range(numIncorrect, numIncorrect + numWith):
        class_out_path = out_path + "/with_mask"
    else:
        class_out_path = out_path + "/without_mask"

    net.setInput(blob)
    detections = net.forward()
    cnt = 0
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
            if x in range(width) and y in range(height) and x1 in range(width) and y1 in range(height):
                img = curr_img[y:y1, x:x1]  # [row, column]
                if img.shape[0] > 0 and img.shape[1] > 0:
                    cv2.imwrite(class_out_path + "/" + str(j) + "_" + str(i) + ".jpg", img)
                    cnt += 1
    if cnt == 0:
        cv2.imwrite(class_out_path + "/" + str(j) + ".jpg", curr_img)

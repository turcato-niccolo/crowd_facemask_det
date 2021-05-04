import cv2


def play(video_cap):
    while cap.isOpened():
        ret, frame = video_cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break


cap = cv2.VideoCapture('output.mp4')
play(cap)
cap.release()
cv2.destroyAllWindows()

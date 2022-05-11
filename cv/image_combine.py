import cv2
import numpy as np

ball_path = "./frames_output/img-0001_ball_without_black.png"
stick_path = "./frames_output/img-0001_stick.png"

img2 = cv2.imread(ball_path)
img1 = cv2.imread(stick_path)

dim = (img1.shape[1], img1.shape[0])
resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

added_image = cv2.addWeighted(img1, 1, resized, 1, 0)
cv2.imwrite("./frames_output/img-0001_final.png", added_image)

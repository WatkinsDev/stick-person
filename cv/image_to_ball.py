# CODE FOR BALL DETECTION

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img_path = "./frames/img-0001.png"

# Inference
results = model(img_path)

# Results
# results.print()
# results.show()
# results.crop()
# results.save()
# results.pandas()

print("type(results)")
print(type(results))

objects_detected = results.pandas().xyxy[0]
print("pandas_val.xyxy[0]")
print(objects_detected)
print("just sports ball")
print(objects_detected[objects_detected.name == "sports ball"])

# Load an color image in grayscale
img = cv2.imread(img_path, 0)
output_path = "./frames_output/img-0001_ball.png"

# show image
x_min = 1818  # 1818.487915  # 50
y_min = 1634  # 1634.062988  # 50
x_max = 2044  # 2044.704834  # 150
y_max = 1859  # 1859.912964  # 150

# contours = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])
contours = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

# cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
# cv2.imwrite(output_path, img)

# stencil = np.zeros(img.shape, 4).astype(img.dtype)
# color = [255, 255, 255, 0]
# cv2.fillPoly(stencil, [contours], color)
# result = cv2.bitwise_and(img, stencil)

stencil = np.zeros((img.shape[0], img.shape[1])).astype(img.dtype)
color = [255, 255, 255]
cv2.fillPoly(stencil, [contours], color)
result = cv2.bitwise_and(img, stencil)

cv2.imwrite(output_path, result)
print("Wrote first ball")

img2 = cv2.imread(output_path)
# threshold on black to make a mask
color = (0, 0, 0)
mask = np.where((img2 == color).all(axis=2), 0, 255).astype(np.uint8)
# put mask into alpha channel
result = img2.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask
# save resulting masked image
cv2.imwrite("./frames_output/img-0001_ball_without_black.png", result)

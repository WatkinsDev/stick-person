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

objects_detected = results.pandas().xyxy[0]
sports_ball = objects_detected[objects_detected.name == "sports ball"]
img = cv2.imread(img_path, 0)

if not sports_ball.empty:

    # show image
    x_min = int(sports_ball.xmin)
    y_min = int(sports_ball.ymin)
    x_max = int(sports_ball.xmax)
    y_max = int(sports_ball.ymax)

    contours = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )

    stencil = np.zeros((img.shape[0], img.shape[1])).astype(img.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [contours], color)
    result = cv2.bitwise_and(img, stencil)
else:
    result = np.zeros_like(img)

output_path = "./frames_output/img-0001_ball.png"
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

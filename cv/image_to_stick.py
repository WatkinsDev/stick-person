# CODE TO GET THE APLHA POSE ESTIMATIONS
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import (
    detector_to_alpha_pose,
    heatmap_to_coord_alpha_pose,
)
import numpy as np

detector = model_zoo.get_model("yolo3_mobilenet1.0_coco", pretrained=True)
pose_net = model_zoo.get_model("alpha_pose_resnet101_v1b_coco", pretrained=True)

# detector.reset_class(["person", "sports ball"], reuse_weights=["person", "sports ball"])
detector.reset_class(["person"], reuse_weights=["person"])

frame_path = "./frames/img-0001.png"
x, img = data.transforms.presets.yolo.load_test(frame_path, short=512)
print("Shape of pre-processed image:", x.shape)

class_IDs, scores, bounding_boxs = detector(x)

pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)

predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

print("bounding_boxs")
print(bounding_boxs)

ax = utils.viz.plot_keypoints(
    # img,
    255 * np.ones_like(img, dtype=np.uint8),
    pred_coords,
    confidence,
    class_IDs,
    bounding_boxs,
    scores,
    # box_thresh=0.5,
    box_thresh=0.99999999999999999999,
    keypoint_thresh=0.2,
)
print("type checking")
print(type(ax))
print(type(img))
plt.show()


# CODE FOR BALL DETECTION

import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "./frames/img-0001.png"

# Inference
results = model(img)
print("type(results)")
print(type(results))

# Results
# results.print()
results.show()
# results.crop()
# results.save()
# results.pandas()
objects_detected = results.pandas().xyxy[0]
print("pandas_val.xyxy[0]")
print(objects_detected)
print("just sports ball")
print(objects_detected[objects_detected.name == "sports ball"])


# CODE FOR READING IMAGES AND WRITING THEM
# from cartoonizer import cartoonize
# import cv2
# import os
# import time

# in_dir = './imgs/input'
# out_dir = './imgs/output'

# os.mkdir(out_dir)

# for f in os.listdir(in_dir):
#     image = cv2.imread(os.path.join(in_dir, f))
#     print('==============')
#     print(f)
#     start_time = time.time()
#     output = cartoonize(image)
#     end_time = time.time()
#     print("time: {0}s".format(end_time-start_time))
#     name = os.path.basename(f)
#     tmp = os.path.splitext(name)
#     name = tmp[0]+"_cartoon" + tmp[1]
#     name = os.path.join(out_dir, name)
#     print("write to {0}".format(name))
#     cv2.imwrite(name, output)

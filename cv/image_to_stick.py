"""2. Predict with pre-trained AlphaPose Estimation models
==========================================================

This article shows how to play with pre-trained Alpha Pose models with only a few
lines of code.

First let's import some necessary libraries:
"""
import cv2
import numpy as np
from email.mime import image
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import (
    detector_to_alpha_pose,
    heatmap_to_coord_alpha_pose,
)

detector = model_zoo.get_model("yolo3_mobilenet1.0_coco", pretrained=True)
pose_net = model_zoo.get_model("alpha_pose_resnet101_v1b_coco", pretrained=True)

detector.reset_class(["person"], reuse_weights=["person"])

x, img = data.transforms.presets.yolo.load_test("./frames/img-0001.png", short=512)
print("Shape of pre-processed image:", x.shape)

class_IDs, scores, bounding_boxs = detector(x)

pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)

predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

ax = utils.viz.plot_keypoints(
    255 * np.ones([x.shape[2], x.shape[3]], np.uint8),
    pred_coords,
    confidence,
    class_IDs,
    bounding_boxs,
    scores,
    box_thresh=0.9999999999999,
    keypoint_thresh=0.2,
)
ax.set_axis_off()
# plt.show()
output_path = "./frames_output/img-0001.png"
plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

# fig.savefig(fname, dpi)

# print("type(ax)")
# print(type(ax))
# # print(x.shape[2])
# # print(x.shape[3])
# # image_q = utils.viz.expand_mask(
# #     pred_coords,
# #     bounding_boxs,
# #     (x.shape[2], x.shape[3]),
# #     scores=scores,
# #     thresh=0.5,
# #     scale=1.0,
# #     sortby=None,
# # )
# # print(type(image_q))
# cv2.imwrite(output_path, ax)

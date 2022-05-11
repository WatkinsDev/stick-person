import os
import torch
from image_combine import image_combine
from image_to_ball import image_to_ball
from image_to_stick import image_to_stick
from gluoncv import model_zoo, data, utils

dir_path = r"./frames"
count = 0

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # or yolov5n - yolov5x6, custom
yolo_old = model_zoo.get_model("yolo3_mobilenet1.0_coco", pretrained=True)
yolo_old.reset_class(["person"], reuse_weights=["person"])

pose_net = model_zoo.get_model("alpha_pose_resnet101_v1b_coco", pretrained=True)


for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
        file_number = str(count).zfill(4)

        print(f"Running process for file #{file_number}")
        image_to_ball(file_number, yolo_model)
        image_to_stick(file_number, detector=yolo_old, pose_net=pose_net)
        image_combine(file_number)
# CODE FOR BALL DETECTION

import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "./frames/img-0001.png"

# Inference
results = model(img)

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

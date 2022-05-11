import cv2
import numpy as np
from matplotlib import pyplot as plt
from gluoncv import data, utils
from gluoncv.data.transforms.pose import (
    detector_to_alpha_pose,
    heatmap_to_coord_alpha_pose,
)


def image_to_stick(file_number, detector, pose_net):
    x, img = data.transforms.presets.yolo.load_test(
        f"./frames/img-{file_number}.png", short=512
    )
    # print("Shape of pre-processed image:", x.shape)

    class_IDs, scores, bounding_boxs = detector(x)
    output_path = f"./frames_wip/img-{file_number}_stick.png"

    try:
        pose_input, upscale_bbox = detector_to_alpha_pose(
            img, class_IDs, scores, bounding_boxs
        )

        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord_alpha_pose(
            predicted_heatmap, upscale_bbox
        )

        ax = utils.viz.plot_keypoints(
            np.ones_like(img),
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
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    except:
        original_path = f"./frames/img-{file_number}.png"
        img = cv2.imread(original_path)
        result = np.zeros_like(img)
        cv2.imwrite(output_path, result)

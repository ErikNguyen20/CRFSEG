import time
import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import math
from itertools import repeat
import sys
import torch
import mmcv, mmdet, mmrotate

from detectron2.structures import RotatedBoxes, BoxMode, Instances
from detectron2.config import *
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



class UnifiedInterface:
    MODEL_CLASSES = ['Right_Index', 'Right_Middle', 'Right_Ring', 'Right_Little', 'Left_Index', 'Left_Middle', 'Left_Ring', 'Left_Little', 'Right_Thumb', 'Left_Thumb']
    FINGER_POSITION_INDEX = {'Right_Thumb': 1, 'Right_Index': 2, 'Right_Middle': 3, 'Right_Ring': 4, 'Right_Little': 5, 
                             'Left_Thumb': 6, 'Left_Index': 7, 'Left_Middle': 8, 'Left_Ring': 9, 'Left_Little': 10}
    EVAL_CSV_COLUMNS = []

    def __init__(self, score_thr=0.6):
        self.score_thr = score_thr

    @staticmethod
    def get_rotated_box_corners(cx, cy, w, h, theta_deg):
        t = -math.radians(theta_deg)
        c, s = math.cos(t), math.sin(t)
        rect = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        # TL, TR, BR, BL

        # rotate then translate
        corners = [(cx + c*x - s*y, cy + s*x + c*y) for x, y in rect]
        return corners

    @staticmethod
    def get_tl_br_from_rotated_box(rotated_corners):
        # Important since there may be no notion of rotation in SLIPSEG
        xs = [x for x, y in rotated_corners]
        ys = [y for x, y in rotated_corners]

        x_tl = min(xs)
        y_tl = min(ys)
        x_br = max(xs)
        y_br = max(ys)
        return x_tl, y_tl, x_br, y_br


    def from_detectron(self, model_output, image_id: str):
        result = []

        out = model_output["instances"].to("cpu")
        out = out.get_fields()
        if len(out.keys()) == 0 or out["scores"] is None or len(out["scores"]) == 0:
            return result
        
        boxes = out['pred_boxes'].tensor
        for i in range(len(boxes)):
            box = boxes[i]
            score = out['scores'][i].item()

            if score < self.score_thr:
                continue

            finger = self.MODEL_CLASSES[out['pred_classes'][i].item()]
            finger_pos = self.FINGER_POSITION_INDEX[finger]

            rotated_corners = self.get_rotated_box_corners(box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item())
            # x_tl, y_tl, x_br, y_br = self.get_tl_br_from_rotated_box(rotated_corners)

            (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = rotated_corners
            result.append({
                "image_id": image_id, "finger_pos": finger_pos, 
                "tl_x": tl_x, "tl_y": tl_y, "tr_x": tr_x, "tr_y": tr_y, "br_x": br_x, "br_y": br_y, "bl_x": bl_x, "bl_y": bl_y
                })
        
        result.sort(key=lambda x: x["finger_pos"])
        return result
    
    def from_mmdet(self, model_output, image_id: str):
        inst = self.mmdet_to_detectron_instance(model_output)
        if inst is None:
            return []
        return self.from_detectron({"instances": inst}, image_id)

    def get_normalized_model_output(self, model_output, image_id: str):
        inst = self.normalize_model_output_to_detectron(model_output)
        return self.from_detectron({"instances": inst}, image_id)

    def mmdet_to_detectron_instance(self, model_output):
        boxes = []
        scores = []
        classes = []
        for class_i in range(len(model_output)):
            if len(model_output[class_i]) == 0 or model_output[class_i].shape[0] == 0:
                continue
            # Note: I just select the first box, but there can be multiple boxes for the class (rare), but if thats the case maybe select the highest score one?
            # box = model_output[class_i][0]
            box = model_output[class_i][np.argmax(model_output[class_i][:, 5])]
            xc, yc, h, w, angle = map(float, box[:5])
            score = box[5].item()

            if score < self.score_thr:
                continue

            boxes.append([xc, yc, w, h, angle])
            scores.append(score)
            classes.append(class_i)
        
        if len(boxes) == 0:
            return None

        inst = Instances((1500, 1600))
        inst.pred_boxes = RotatedBoxes(torch.tensor(boxes, dtype=torch.float32))
        inst.scores = torch.tensor(scores, dtype=torch.float32)
        inst.pred_classes = torch.tensor(classes, dtype=torch.int64)
        return inst

    def normalize_model_output_to_detectron(self, model_output):
        outputs = model_output
        if isinstance(model_output, dict) and "instances" in model_output:
            outputs = model_output["instances"].to("cpu")
        elif isinstance(model_output, Instances):
            outputs = model_output.to("cpu")
        elif isinstance(model_output, list):
            outputs = self.mmdet_to_detectron_instance(model_output).to("cpu")
        else:
            raise ValueError("Invalid model output type")
        return outputs

    def visualize_output_img(self, img, model_output, scale=1.0):
        outputs = self.normalize_model_output_to_detectron(model_output)

        MetadataCatalog.get('Test').set(thing_classes=self.MODEL_CLASSES)
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("Test"), scale=scale)

        out = v.draw_instance_predictions(outputs)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()

    def append_to_csv(self, csv_file, result):
        # Create CSV with header
        if not os.path.exists(csv_file):
            pd.DataFrame(columns=self.EVAL_CSV_COLUMNS).to_csv(csv_file, index=False, encoding="utf-8")
        
        pd.DataFrame(result, columns=self.EVAL_CSV_COLUMNS).to_csv(csv_file, mode="a", header=False, index=False, encoding="utf-8")


def evaluate_on_CRFSEG():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join('crfseg/trained_model')

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #this is far lower than usual.  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0013499.pth")
    cfg.MODEL.DEVICE = "cpu"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold value to filter our low scoring bboxes
    predictor = DefaultPredictor(cfg)


    imname = 'crfseg/test_image/Image_02_1_2.bmp'
    img = mmcv.imread(imname)
    img = rotate_img(img, -15)

    start_time = time.perf_counter()
    outputs = predictor(img)  
    elapsed = time.perf_counter() - start_time
    print(f"Inference time (1 image): {elapsed:.4f} seconds")

    interface = UnifiedInterface(score_thr=0.6)
    out_dict = interface.get_normalized_model_output(outputs, "testlol")
    print(out_dict)
    interface.visualize_output_img(img, outputs)
    imshow(img, boxes=out_dict)


def evaluate_on_TransSEG():
    config_file = 'transSEG/roi_trans_swin_tiny_fpn_1x_dota_le90_trained.py'
    checkpoint_file = 'transSEG/trained_model/transSEG.pth'

    cfg = Config.fromfile(config_file)
    cfg.model.roi_head.bbox_head[0].num_classes = 10
    cfg.model.roi_head.bbox_head[1].num_classes = 10
    model = init_detector(cfg, checkpoint_file, device='cpu')
    model.eval()
    model.cfg = cfg

    if not hasattr(np, "int0"):
        # Quick fix for a weird problem with numpy in the show results method
        np.int0 = np.int32

    imname = 'crfseg/test_image/Image_02_1_2.bmp'
    img = mmcv.imread(imname)
    # img = rotate_img(img, -15)

    start_time = time.perf_counter()
    result = inference_detector(model, img)
    elapsed = time.perf_counter() - start_time
    print(f"Inference time (1 image): {elapsed:.4f} seconds")

    interface = UnifiedInterface(score_thr=0.6)
    out_dict = interface.get_normalized_model_output(result, "testlol")
    print(out_dict)
    interface.visualize_output_img(img, result)
    imshow(img, boxes=out_dict)

    # show_result_pyplot(model, img, result, score_thr=0.6)


def imread_gray(img_path, ppi, desired_ppi = None):
    # Parse dimensions from name
    base = os.path.basename(img_path)
    name, *_ = base.split(".", 1)

    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename does not contain enough '_' parts: {base}")
    try:
        w = int(parts[-2])
        h = int(parts[-1])
    except ValueError:
        raise ValueError(f"Last two '_' parts are not integers: {base}")

    img = np.fromfile(img_path, dtype=np.uint8)

    # Validate expected size for 8-bit grayscale
    expected = w * h
    if img.size != expected:
        raise ValueError(f"File size mismatch for {base}. got {img.size} pixels, expected {expected} (w={w}, h={h}).")
    img = img.reshape((h, w))

    # img = img[::-1, :]  # This flip bottom up if coordinate system is different

    # PPI-based resample
    if ppi is not None and desired_ppi is not None and ppi > 0 and desired_ppi > 0:
        scale = float(desired_ppi) / float(ppi)
        if abs(scale - 1.0) > 1e-12:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if scale < 1.0:
                interp = cv2.INTER_AREA
            else:
                interp = cv2.INTER_CUBIC

            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def rotate_img(img, degrees):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Clockwise rotation = negative angle
    M = cv2.getRotationMatrix2D(center, -degrees, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated


def imshow(img, ppi=None, boxes=None):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = img.shape[:2]

    if ppi is not None and ppi > 0:
        fig, ax = plt.subplots(figsize=(w / ppi, h / ppi), dpi=ppi)
    else:
        fig, ax = plt.subplots()

    ax.imshow(img)

    if boxes is not None:
        for box in boxes:
            pts = np.array([
                [box["tl_x"], box["tl_y"]],
                [box["tr_x"], box["tr_y"]],
                [box["br_x"], box["br_y"]],
                [box["bl_x"], box["bl_y"]],
            ], dtype=float)

            poly = Polygon(pts, closed=True, fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(poly)
    plt.show()



if __name__ == "__main__":
    evaluate_on_CRFSEG()
    # evaluate_on_TransSEG()
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler


def get_cfg_from_file(cfg_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(cfg_path)
    return cfg


class SemanticSegmentationPredictor(object):
    def __init__(self, cfg):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.predictor = DefaultPredictor(cfg)

    def preprocess_image(self, image, target_size=(1024, 512)):
        h = image.shape[0]
        w = image.shape[1]
        ratio_h = target_size[0] / image.shape[0]
        ratio_w = target_size[1] / image.shape[1]
        if ratio_h > ratio_w:
            expected_size = (int(h * ratio_w), target_size[1])
            padding_y = int((target_size[0] - expected_size[0]) / 2)
            padding_x = 0
        elif ratio_h < ratio_w:
            expected_size = (target_size[0], int(w * ratio_h))
            padding_y = 0
            padding_x = int((target_size[1] - expected_size[1]) / 2)
        else:
            expected_size = target_size
            padding_y = 0
            padding_x = 0
        image = cv2.resize(image, (expected_size[1], expected_size[0]), interpolation=cv2.INTER_LINEAR)
        image_new = np.zeros((target_size[0], target_size[1], 3), dtype="uint8")
        image_new[padding_y:padding_y + image.shape[0], padding_x:padding_x + image.shape[1], :] = image
        return image_new, (padding_y, padding_x), (h, w), expected_size

    def post_process(self, mask, padding_yx, hw, expected_size):
        with torch.no_grad():
            mask = mask[:, padding_yx[0]:padding_yx[0] + expected_size[0], padding_yx[1]: padding_yx[1] + expected_size[1]]
            mask = torch.unsqueeze(mask, dim=0)
            mask = F.interpolate(mask, size=hw, mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask.cpu().numpy()
            mask = np.squeeze(mask, 0)
        return mask

    def run_on_image(self, image):
        image = image[:, :, ::-1]
        image_new, padding_yx, hw, expected_size = self.preprocess_image(image)
        predictions = self.predictor(image_new)
        mask = predictions['sem_seg']
        return self.post_process(mask, padding_yx, hw, expected_size)


if __name__ == '__main__':
    cfg = get_cfg_from_file("/home/jiangwang/code/detectron2/projects/DeepLab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16_coco.yaml")
    cfg.MODEL.WEIGHTS = "/home/jiangwang/code/detectron2/projects/DeepLab/output/model_0019999.pth"

    image = cv2.imread("/home/jiangwang/code/ailab/VirtualStage/KinectMaskGenerator/build/output_frames/0045_test.png")

    predictor = SemanticSegmentationPredictor(cfg)

    results = predictor.run_on_image(image)
    mask_image = (results[1, :, :] * 255).astype('uint8')
    cv2.imwrite('mask.png', mask_image)

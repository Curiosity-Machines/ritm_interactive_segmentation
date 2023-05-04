import argparse

import torch
import numpy as np
import cv2

from isegm.utils import exp
from isegm.inference import utils
from isegm.inference.clicker import Clicker, Click
from interactive_demo.app import InteractiveDemoApp
from isegm.inference.transforms import ZoomIn
from isegm.model.is_hrnet_model import HRNetModel
from isegm.inference.predictors.base import BasePredictor

# python3 demo.py --checkpoint=coco_lvis_h18_itermask --gpu=0

def main():
    print("hello")
    device = "mps"
    state_dict = torch.load('weights/coco_lvis_h18_itermask.pth', map_location='mps')
    model = HRNetModel(width=18, ocr_width=64, small=False, with_aux_output=True, use_leaky_relu=True, use_rgb_conv=True, use_disks=True, norm_radius=5, with_prev_mask=True, cpu_dist_maps=False)
    model.to(device)
    image = cv2.cvtColor(cv2.imread("test.jpg"), cv2.COLOR_BGR2RGB)

    # Load into a MPSFloatType numpy tensor
    result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    clicker = Clicker()
    object_count = 0
    states = []
    probs_history = []

    for param in model.parameters():
        param.requires_grad = False

    # Reset object
    states = []
    probs_history = []
    clicker.reset_clicks()
    zoom_in = ZoomIn(skip_clicks=-1, target_size=(400, 400), expansion_ratio=1.4)
    predictor = BasePredictor(model, device=device, optimize_after_n_clicks=1, net_clicks_limit=None, max_size=800, zoom_in=zoom_in, with_flip=True)
    predictor.set_input_image(image)
    init_mask = None
    clicker.click_indx_offset = 0

    # Add click
    click = Click(is_positive=True, coords=(100, 100))

    pred = predictor.get_prediction(clicker, prev_mask=init_mask)

    if init_mask is not None and len(clicker) == 1:
        pred = predictor.get_prediction(clicker, prev_mask=init_mask)

    torch.cuda.empty_cache()

    if probs_history:
        probs_history.append((self.probs_history[-1][0], pred))
    else:
        probs_history.append((np.zeros_like(pred), pred))

    print(pred)

    # model_cfg = edict()
    #model_cfg.crop_size = (320, 480)
    #model_cfg.num_max_points = 24

    #model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))

    model.to("mps")

if __name__ == '__main__':
    main()

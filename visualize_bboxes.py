import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2_repo import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def visualize_bboxes(dataset_root, bboxes_dir, name=None, num=None, return_images=False):
    
    # cfg = get_cfg()
    # Need only to get the config file for dataset
    # MODEL_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    # cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    # print(cfg.DATASETS.TRAIN[0])
    
    bboxes_list = [i for i in os.listdir(bboxes_dir) if i.endswith('.pt')]

    if name is None:

        if num is None:
            num = len(bboxes_list)
        
        pt_files = np.random.choice(bboxes_list, size=num, replace=False)
    
    else:
        if not name.endswith('.pt'):
            name = name.split('.')[0] + '.pt'
        pt_files = [name]
    
    for pt_filename in pt_files:
        img_name = pt_filename.split('.')[0] + '.jpg'
        scene = ''
        if img_name[0] == '0':
            scene += 'Home_'
        else:
            scene += 'Office_'

        scene += img_name[1:4]

        img_path = os.path.join(dataset_root, scene+'_1', 'jpg_rgb', img_name)

        if not os.path.isfile(img_path):
            img_path = os.path.join(dataset_root, scene+'_2', 'jpg_rgb', img_name)

        im = cv2.imread(img_path)
        outputs = torch.load(os.path.join(bboxes_dir, pt_filename))
        
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get('coco_2017_train'), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        if return_images:
            return v.get_image()

        plt.figure(figsize=(30, 18))
        plt.imshow(v.get_image())
        plt.show()
        plt.close()

        # cv2.imshow(pt_filename, v.get_image()[:,:,::-1])
        # Wait until window close
        # while cv2.getWindowProperty(pt_filename, 0) >= 0:
            # print(f'Entered loop')
            # keyCode = cv2.waitKey(50)
            # print(f'key = {keyCode}')
        
        # cv2.destroyAllWindows()

# cv2.imwrite(f'o2_rpn_{cfg.MODEL.RPN.NMS_THRESH}_roi_{cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}_detperimg_{cfg.TEST.DETECTIONS_PER_IMAGE}_roiscore_{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}.jpg', 
#   v.get_image()[:,:,::-1])

if __name__ == '__main__':
    visualize_bboxes('/mnt/sda2/workspace/DATASETS/ActiveVision', 
        '/mnt/sda2/workspace/self_supervised_outputs/mask_rcnn_R_101_FPN_3x_1_270346/predictor',
        num=5)
import os
import torch
import json
import pickle
import numpy as np
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class ActiveVisionTriplet(Dataset):

    SCENES = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1',
              'Home_003_2', 'Home_004_1', 'Home_004_2', 'Home_005_1',
              'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
              'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1',
              'Home_014_2', 'Home_015_1', 'Home_016_1', 'Office_001_1']

    def __init__(self, dataset_root, triplet_root, instance, image_size,
                 triplet_image_size, get_labels=False):
        self.dataset_root = dataset_root
        self.triplet_root = triplet_root

        if instance in self.SCENES:
            self.img_folder_map = defaultdict(lambda: instance)
        elif instance == 'instance1':
            with open(os.path.join(dataset_root, 'coco_annotations',
                                   'instances_set_1_train.json')) as f:
                self.img_folder_map = json.load(f)['img_folder_map']
        else:
            raise ValueError(f'Invalid instance: {instance}')

        self.image_size = image_size
        self.triplet_image_size = triplet_image_size
        self.get_labels = get_labels

        if self.get_labels:
            # with open(os.path.join(self.dataset_root, 'activevision_label_map.pbtxt'), 'rb') as f:
            #     self.label_map = json.load(f)

            if instance not in self.SCENES:
                raise NotImplementedError('Fetching labels for multiple folder not implemented')

            with open(os.path.join(self.dataset_root, instance, 'annotations.json')) as f:
                self.annotations = json.load(f)

        # TODO: Make this a parameter
        self.transforms = transforms.Compose([
            transforms.Resize(self.triplet_image_size),
            transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                        std=(0.5, 0.5, 0.5))])

        self.pickle_names = [i for i in sorted(os.listdir(self.triplet_root)) if
                             i.endswith('pickle')]
        assert len(self.pickle_names) > 0, f'No pickle file found in {self.triplet_root}'
        self.pickles_dict = {}
        self.triplets = []

        for pickle_name in self.pickle_names:
            img_name = pickle_name.split('.')[0] + '.jpg'

            with open(os.path.join(triplet_root, pickle_name), 'rb') as f:
                content = pickle.load(f)
                self.pickles_dict[img_name] = content

            for neighbor_img_name, neighbor_triplets in content.items():
                for neighbor_triplet in neighbor_triplets:
                    triplet_dict = {
                        'ref': [img_name, neighbor_triplet[0]],
                        'pos': [neighbor_img_name, neighbor_triplet[1]],
                        'neg': [img_name, neighbor_triplet[2]]
                    }
                    self.triplets.append(triplet_dict)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Labels only calculated if flag is set
        ref_crop, pos_crop, neg_crop, labels = self.get_triplet_img(idx)

        return self.transforms(ref_crop), self.transforms(
            pos_crop), self.transforms(neg_crop), labels

    def get_triplet_img(self, idx):
        triplet = self.triplets[idx]
        # print(idx)
        ref_img_name = triplet['ref'][0]
        pos_img_name = triplet['pos'][0]

        ref_img_folder = self.img_folder_map[ref_img_name]
        pos_img_folder = self.img_folder_map[pos_img_name]

        ref_bbox = triplet['ref'][1]
        pos_bbox = triplet['pos'][1]
        neg_bbox = triplet['neg'][1]

        ref_img = Image.open(os.path.join(self.dataset_root, ref_img_folder,
                                          'jpg_rgb',
                                          triplet['ref'][0])).resize(
            self.image_size)
        pos_img = Image.open(os.path.join(self.dataset_root, pos_img_folder,
                                          'jpg_rgb',
                                          triplet['pos'][0])).resize(
            self.image_size)
        # ref_crop = ref_img.crop(ref_bbox).resize(self.triplet_image_size)
        # pos_crop = pos_img.crop(pos_bbox).resize(self.triplet_image_size)
        # neg_crop = ref_img.crop(neg_bbox).resize(self.triplet_image_size)

        ref_crop = ref_img.crop(ref_bbox)
        pos_crop = pos_img.crop(pos_bbox)
        neg_crop = ref_img.crop(neg_bbox)

        labels = None

        if self.get_labels:
            labels = []

            annotatations = self.annotations
            # TODO: Make this a parameter
            scale_img_org_x = 1920/self.image_size[0]
            scale_img_org_y = 1080/self.image_size[1]

            # TODO: Make neg bbox independent of ref image
            for bbox, img_name in [[ref_bbox, ref_img_name], [pos_bbox, pos_img_name], [neg_bbox, ref_img_name]]:

                bbox = bbox.copy()
                # print(idx, img_name, bbox)
                # Bring to original image scale
                bbox[0::2] *= scale_img_org_x
                bbox[1::2] *= scale_img_org_y
                img_annotation = np.array(annotatations[img_name]['bounding_boxes'])
                matched_bbox_idx = self.iou(ref_bbox, img_annotation)
                img_label = 0

                if matched_bbox_idx is not None:
                    img_label = img_annotation[matched_bbox_idx][4]

                labels.append(img_label)

        return ref_crop, pos_crop, neg_crop, labels

    @staticmethod
    def iou(bbox1, bboxes2):

        if len(bboxes2) == 0:
            return None

        iou_list = []

        area_bbox1 = (bbox1[3]-bbox1[1]) * (bbox1[2]-bbox1[0])

        for bbox2 in bboxes2:
            intersection_x1 = max(bbox1[0], bbox2[0])
            intersection_y1 = max(bbox1[1], bbox2[1])
            intersection_x2 = min(bbox1[2], bbox2[2])
            intersection_y2 = min(bbox1[3], bbox2[3])

            area_bbox2 = (bbox2[3]-bbox2[1]) * (bbox2[2]-bbox2[0])
            intersect_area = max(intersection_x2-intersection_x1, 0)*max(intersection_y2-intersection_y1, 0)

            iou = intersect_area/(area_bbox1+area_bbox2-intersect_area)
            iou_list.append(iou)

        max_iou_idx = np.argmax(iou_list)
        max_iou = iou_list[max_iou_idx]

        if max_iou > 0:
            pass

        # TODO: Make this a parameter
        if max_iou >= 0.1:
            return max_iou_idx

        return None

    def visualize_triplet(self, idx):
        import matplotlib.pyplot as plt
        from matplotlib import patches
        triplet = self.triplets[idx]

        for k, v in triplet.items():
            print(k, v)

        ref_img_folder = self.img_folder_map[triplet['ref'][0]]
        pos_img_folder = self.img_folder_map[triplet['pos'][0]]

        ref_bbox = triplet['ref'][1]
        pos_bbox = triplet['pos'][1]
        neg_bbox = triplet['neg'][1]

        # TODO: Make negative image independent of ref image
        ref_img = Image.open(os.path.join(self.dataset_root, ref_img_folder,
                                          'jpg_rgb',
                                          triplet['ref'][0])).resize(
            self.image_size)
        pos_img = Image.open(os.path.join(self.dataset_root, pos_img_folder,
                                          'jpg_rgb',
                                          triplet['pos'][0])).resize(
            self.image_size)

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(25, 14))
        
        # raise Exception()
        ax1.imshow(ref_img)
        ax2.imshow(pos_img)

        ref_rect = patches.Rectangle((ref_bbox[0], ref_bbox[1]),
                                     ref_bbox[2] - ref_bbox[0],
                                     ref_bbox[3] - ref_bbox[1],
                                     linewidth=3, edgecolor='b',
                                     facecolor='none')
        ax1.add_patch(ref_rect)
        ax1.text(ref_bbox[0], ref_bbox[1], 'ref', fontsize=10)

        pos_rect = patches.Rectangle((pos_bbox[0], pos_bbox[1]),
                                     pos_bbox[2] - pos_bbox[0],
                                     pos_bbox[3] - pos_bbox[1],
                                     linewidth=3, edgecolor='b',
                                     facecolor='none')
        ax2.add_patch(pos_rect)
        ax2.text(pos_bbox[0], pos_bbox[1], 'pos', fontsize=10)

        neg_rect = patches.Rectangle((neg_bbox[0], neg_bbox[1]),
                                     neg_bbox[2] - neg_bbox[0],
                                     neg_bbox[3] - neg_bbox[1],
                                     linewidth=3, edgecolor='r',
                                     facecolor='none')
        ax1.add_patch(neg_rect)
        ax1.text(neg_bbox[0], neg_bbox[1], 'neg', fontsize=10)
        plt.show()

    def visualize_multi_triplets(self, num=None, random=True):
        import numpy as np

        num_triplets = len(self.triplets)

        if num is None:
            num = num_triplets
            print(f'Showing all triplets = {num}')

        all_indices = list(range(num_triplets))

        if random:
            all_indices = np.random.choice(all_indices, num_triplets,
                                           replace=False)

        for idx in all_indices[:num]:
            print(f'Index = {idx}')
            self.visualize_triplet(idx)


if __name__ == '__main__':
    # a = ActiveVisionTriplet('/mnt/sda2/workspace/DATASETS/ActiveVision',
    #                         '/home/sulabh/workspace-ubuntu/triplets',
    #                         instance='instance1', image_size=(1333, 750))
    # a = ActiveVisionTriplet('/mnt/sda2/workspace/DATASETS/ActiveVision/',
    #                     '/home/sulabh/workspace-ubuntu/triplet_nodes/Home_002_1/train',
    #                     instance='Home_002_1',
    #                     image_size=(1333, 750),
    #                     triplet_image_size=(224, 224), get_labels=True)

    a = ActiveVisionTriplet('/mnt/sda2/workspace/DATASETS/ActiveVision/',
                        '/mnt/sda2/workspace/self_supervised_outputs/triplets/mask_rcnn_R_101_FPN_3x_1_270346/Home_003_2/triplets',
                        instance='Home_003_2',
                        image_size=(1920, 1080),
                        # image_size=(1344, 768),
                        triplet_image_size=(224, 224), get_labels=True)

    a.visualize_multi_triplets(num=5, random=True)
    # for i in range(2280, 2300):
    #     a.visualize_triplet(i)

    # ref, pos, neg, l = a.get_triplet_img(100)
    # print(l)
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    # ax1.imshow(ref)
    # ax2.imshow(pos)
    # ax3.imshow(neg)
    # plt.show()

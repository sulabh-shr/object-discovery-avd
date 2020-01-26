import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle


class BboxEvaluator:
    SCENES = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1',
              'Home_003_2', 'Home_004_1', 'Home_004_2', 'Home_005_1',
              'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
              'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1',
              'Home_014_2', 'Home_015_1', 'Home_016_1', 'Office_001_1']

    def __init__(self, dataset_root, proposals_path, evaluation_params):
        self.proposals_path = proposals_path
        self.dataset_root = dataset_root
        self.evaluation_params = evaluation_params

        self.ann = {}  # Annotation of all images in Active Vision format
        self.img_folder_map = {}  # Image to Scene mapping

        self.create_folder_map()

        self.matched = None
        self.total = None

    def create_folder_map(self):
        """Create a mapping of image to folder and load annotation files.
        """
        for scene in self.SCENES:
            scene_path = os.path.join(self.dataset_root, scene)
            scene_img_path = os.path.join(scene_path, 'jpg_rgb')
            images = os.listdir(scene_img_path)

            for img in images:
                self.img_folder_map[img] = scene

            ann_path = os.path.join(scene_path, 'annotations.json')

            with open(ann_path, 'r') as f:
                self.ann.update(json.load(f))

    def load_pt_file(self, filename):
        if not filename.endswith('.pt'):
            filename = filename.split('.')[0] + '.pt'
        try:
            outputs = torch.load(os.path.join(self.proposals_path, filename))

        # Get the bounding boxes
        except FileNotFoundError:
            print(f'File {filename} not found!!')
            return None

        return outputs

    def calculate_recall(self):
        print(f'Starting recall calculation for {self.proposals_path}...')

        num_matched_boxes = defaultdict(lambda: defaultdict(int)) # matched boxes per instance per scene
        num_gt_boxes = defaultdict(lambda: defaultdict(int))      # total boxes per instance per scene
        proposal_files = [i for i in os.listdir(self.proposals_path) if (i.endswith('.pt'))]

        # TODO: Make file selection a param
        for idx, proposal_file in tqdm(enumerate(proposal_files), total=len(proposal_files)):

            img_file = proposal_file.split('.')[0] + '.jpg'

            outputs = self.load_pt_file(proposal_file)
            proposal_bboxes = outputs['instances'].pred_boxes[:self.evaluation_params['proposals_per_img']]
            gt_bboxes = self.ann[img_file]['bounding_boxes']

            # Recall does not change if no ground truth bounding box
            if len(gt_bboxes) == 0:
                continue

            scene = self.img_folder_map[img_file]

            for gt_bbox in gt_bboxes:

                label = gt_bbox[4]

                num_gt_boxes[scene][label] += 1
                gt_bbox_area = (gt_bbox[3] - gt_bbox[1]) * (gt_bbox[2] - gt_bbox[0])

                # Iterate over proposals
                for proposal_bbox in proposal_bboxes:

                    intersect_xmin = max(proposal_bbox[0], gt_bbox[0])
                    intersect_ymin = max(proposal_bbox[1], gt_bbox[1])
                    intersect_xmax = min(proposal_bbox[2], gt_bbox[2])
                    intersect_ymax = min(proposal_bbox[3], gt_bbox[3])
                    intersect_area = max(0, (intersect_xmax - intersect_xmin)) * max((intersect_ymax - intersect_ymin), 0)

                    proposal_bbox_area = (proposal_bbox[3] - proposal_bbox[1]) * (proposal_bbox[2] - proposal_bbox[0])
                    union_area = gt_bbox_area + proposal_bbox_area - intersect_area
                    iou = intersect_area / union_area

                    # Break after first match because recall only needs 1
                    if iou >= self.evaluation_params['iou_thresh']:
                        num_matched_boxes[scene][label] += 1

                        break

                # Make sure there is 0 value if No match at all
                num_matched_boxes[scene][label] += 0

        self.matched = num_matched_boxes
        self.total = num_gt_boxes

        return num_matched_boxes, num_gt_boxes

    def report(self, save=True):
        print('Reporting result...')
        scenes = sorted(list(self.matched.keys()))

        report_str = ""
        report_str += f'\nProposals path : {self.proposals_path}'
        report_str += f'\nUsing params   : \n{self.evaluation_params}\n'
        report_str += '\n' + '-'*70 + '\n'

        for scene in scenes:
            report_str += f'\nScene = {scene}'
            available_labels = sorted(list(self.total[scene].keys()))

            scene_match_bboxes = 0
            scene_gt_bboxes = 0

            for label in available_labels:
                label_matched = self.matched[scene][label]
                label_total = self.total[scene][label]

                scene_match_bboxes += label_matched
                scene_gt_bboxes += label_total

                report_str += f'\nFor LABEL = {label:2d} | Recall = {label_matched/label_total:.3f}| '
                report_str += f'Total = {label_total:<4d} | Matched = {label_matched:<4d}'

            report_str += f'\nTotal Recall = {scene_match_bboxes/scene_gt_bboxes:.3f}'
            report_str += '\n' + '-'*70 + '\n'

        print(report_str)
        
        if save:
            report_path = os.path.join(self.proposals_path, '..', 'recall.txt')
            with open(report_path, 'w') as f:
                f.write(report_str)
            print('Saved report at {report_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--proposals_path", type=str, required=True, help="Path to where torch outputs are stored.")
    args = parser.parse_args()

    DATASET_PATH = args.dataset_root
    PROPOSALS_PATH = args.proposals_path

    EVALUATION_PARAMS = {
        'proposals_per_img': 100,
        'iou_thresh': 0.5
    }

    BE = BboxEvaluator(dataset_root=DATASET_PATH, proposals_path=PROPOSALS_PATH, 
        evaluation_params=EVALUATION_PARAMS)

    BE.calculate_recall()
    BE.report()


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict

# import json_tricks as json
import ujson as json
import numpy as np

from dataset.JointsDataset import JointsDataset
from .coco import COCODataset
from .mpii import MPIIDataset
from .posetrack import PoseTrackDataset


logger = logging.getLogger(__name__)


class AIFIDataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None, pose_format='mpii'):
        super(AIFIDataset, self).__init__(cfg, root, image_set, is_train, transform)

        format_dataset = {
            'coco': COCODataset,
            'mpii': MPIIDataset,
            'posetrack': PoseTrackDataset
        }[pose_format]
        self.num_joints = getattr(format_dataset, 'num_joints')
        self.flip_pairs = getattr(format_dataset, 'flip_pairs')
        self.upper_body_ids = getattr(format_dataset, 'upper_body_ids')
        self.lower_body_ids = getattr(format_dataset, 'lower_body_ids')

        self.parent_ids = None
        self.pixel_std = 200

        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        # load annotations
        self.all_images_dict = self._load_annotations()
        logger.info('=> num_images: {}'.format(len(self.all_images_dict)))

        self.db = self._get_db()
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _load_annotations(self):
        image_names_file = os.path.join(self.root, 'coco_annotations', 'image_names.json')
        with open(image_names_file, 'r') as f:
            raw_annotation = json.load(f)
            image_names = raw_annotation['images']

        all_images_dict = {}
        for image in image_names:
            all_images_dict[image['id']] = image
        return all_images_dict

    def _get_db(self):
        if self.use_gt_bbox:
            raise NotImplementedError
        else:
            return self._load_coco_person_detection_results()

    def _load_coco_person_detection_results(self):
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        image_root = os.path.join(self.root, 'frames')
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue

            image_id = det_res['image_id']
            image = self.all_images_dict[image_id]
            image_path = os.path.join(image_root, image['file_name'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': image_path,
                'image_id': image['file_name'],
                'bbox_tlwh': np.asarray(box),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    @staticmethod
    def read_mot_results(filename, is_gt, is_ignore):
        valid_labels = {1}
        ignore_labels = {2, 7, 8, 12}
        results_dict = dict()
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in f.readlines():
                    linelist = line.split(',')
                    if len(linelist) < 7:
                        continue
                    fid = int(linelist[0])
                    if fid < 1:
                        continue
                    results_dict.setdefault(fid, list())

                    if is_gt:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            mark = int(float(linelist[6]))
                            if mark == 0 or label not in valid_labels:
                                continue
                        score = 1
                    elif is_ignore:
                        if 'MOT16-' in filename or 'MOT17-' in filename:
                            label = int(float(linelist[7]))
                            vis_ratio = float(linelist[8])
                            if label not in ignore_labels and vis_ratio >= 0:
                                continue
                        else:
                            continue
                        score = 1
                    else:
                        score = float(linelist[6])

                    tlwh = tuple(map(float, linelist[2:6]))
                    target_id = int(linelist[1])

                    results_dict[fid].append((tlwh, target_id, score))

        return results_dict

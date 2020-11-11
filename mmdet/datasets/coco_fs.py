import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.core import eval_recalls
from mmdet.utils import print_log
from .custom import CustomDataset
from .registry import DATASETS
from .coco import CocoDataset
from mmdet.core import eval_map, eval_recalls

@DATASETS.register_module
class CocoDatasetFs(CocoDataset):

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    #            'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    #            'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    #            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    CLASSES = ('truck', 'skateboard', 'banana', 'stop sign', 'parking meter',
               'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
               'potted plant', 'sheep', 'couch', 'train', 'tv',
               )

    # NOVEL_CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    #                  'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
    #                  'potted plant', 'sheep', 'couch', 'train', 'tv',)

    novel_sets = [['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                   'potted plant', 'sheep', 'couch', 'train', 'tv', ]]
    # DS_NAME = 'coco_n20_40'
    DS_NAME = 'coco_40'
    # DS_NAME = 'COCO_b20'
    # DS_NAME = 'COCO_n20'
    def _filter_imgs(self, min_size=32):
        # valid_inds = []
        # ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # """Filter images too small or without ground truths."""
        # for i, img_info in enumerate(self.img_infos):
        #     if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 81
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 81
            for l in labels:
                if have_label_list[l] + temp_list[l] >= self.n_shot:
                    have_label_set = set()
                    break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0] * 81
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds

    def evaluatebak(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 # results,
                 # metric='mAP',
                 # logger=None,
                 # proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            # if self.year == 2007:
            #     ds_name = 'voc07'
            # else:
            #     ds_name = self.dataset.CLASSES
            ds_name = self.DS_NAME
            mean_ap, all_results = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            # novel_sets_mAP = []
            for i, ns in enumerate(self.novel_sets):
                ns_ap = [all_results[self.CLASSES.index(c)]['ap'] for c in ns]
                ns_mAP = np.array(ns_ap).mean()
                print(('| novel_set' + str(i + 1)).ljust(14) + '| mAP : ' +
                      str(np.around(ns_mAP, 3)).ljust(5, '0') + ' |')
            print('+-------------+-----+-------+')
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i,
                                                                            j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results

@DATASETS.register_module
class CocoDataset30s(CocoDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 30
        super(CocoDataset30s, self).__init__(**kwargs)

@DATASETS.register_module
class CocoDataset10s(CocoDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 10
        super(CocoDataset10s, self).__init__(**kwargs)

@DATASETS.register_module
class CocoDataset5s(CocoDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 5
        super(CocoDataset5s, self).__init__(**kwargs)

@DATASETS.register_module
class CocoDataset1s(CocoDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 1
        super(CocoDataset1s, self).__init__(**kwargs)

@DATASETS.register_module
class CocoDatasetBase(CocoDataset):
    CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
               'potted plant', 'sheep', 'couch', 'train', 'tv',
               'truck', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    NOVEL_CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                     'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                     'potted plant', 'sheep', 'couch', 'train', 'tv',)

    def _filter_imgs(self, min_size=5):
        # valid_inds = []
        # ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # """Filter images too small or without ground truths."""
        # for i, img_info in enumerate(self.img_infos):
        #     if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 81
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            # print(i, ' : ', labels)
            have_label_set = set()
            temp_list = [0] * 81
            for l in labels:
                if self.CLASSES[l-1] in self.NOVEL_CLASSES:
                    have_label_set = set()
                    break
                # if have_label_list[l] + temp_list[l] >= 1:
                #     have_label_set = set()
                #     break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0] * 81
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds


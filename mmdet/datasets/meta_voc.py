from mmdet.core import eval_map, eval_recalls
from .registry import DATASETS
from .xml_style import XMLDataset
import numpy as np

from mmdet.utils import print_log
import os.path as osp
import os
import xml.etree.ElementTree as ET

import random
import mmcv
from .pipelines import Compose

from mmdet.datasets.pipelines.formating import to_tensor
import torch
import mmcv.image as mi
from mmdet.datasets.pipelines.loading import LoadImageFromFile, LoadAnnotations
from mmdet.datasets.pipelines.transforms import Resize

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
support_pipeline = [
    # dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
]

@DATASETS.register_module
class MetaVOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    base_sets = [
        ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person',
         'pottedplant', 'sheep', 'train', 'tvmonitor'],
        ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person',
         'pottedplant', 'sheep', 'train', 'tvmonitor'],
        ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
         'person', 'pottedplant', 'train', 'tvmonitor']
    ]

    filter_classes_dict = {
        'base1': base_sets[0],
        'base2': base_sets[1],
        'base3': base_sets[2],
        'novel1': novel_sets[0],
        'novel2': novel_sets[1],
        'novel3': novel_sets[2],
    }

    def __init__(self,
                 enable_ignore=True,
                 process_classes=None,
                 instance_path=None,
                 num_support_instances=5,
                 pre_test=False,
                 **kwargs):
        if process_classes is not None:
            assert isinstance(process_classes, tuple) and len(process_classes)==2
            assert process_classes[0] in self.filter_classes_dict.keys()
            assert process_classes[1] in ['filter', 'retain']
            self.process_classes = self.filter_classes_dict[process_classes[0]]
            self.process_type = process_classes[1]
        else:
            self.process_classes = None
            self.process_type = None
        self.enable_ignore = enable_ignore
        self.instance_path = instance_path
        self.pre_test = pre_test

        super(MetaVOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

        self.instance_files_dict = dict()
        if instance_path is not None:
            image_set = self.ann_file.split('/')[-1].split('.')[0]
            if 'shot' in image_set:
                image_set = image_set[:-15]
            self.instance_path = osp.join(self.instance_path, image_set)
            for class_name in self.CLASSES:
                instance_dir = osp.join(self.instance_path, class_name)
                l = list()
                for root, dirs, files in os.walk(instance_dir):
                    for file in files:
                        if file.endswith('.jpg'):
                            l.append(osp.join(instance_dir, file))
                self.instance_files_dict[class_name] = l
        self.num_support_instances = num_support_instances
        self.load_img = LoadImageFromFile()
        self.load_anno = LoadAnnotations(with_bbox=True)
        # self.resize_multiscale = Resize(img_scale=[(1000, 600), (1000, 60)], multiscale_mode='range', keep_ratio=True)
        self.resize_multiscale = Resize(img_scale=[(1000, 600), (1000, 60)], multiscale_mode='range', keep_ratio=True)
        self.support_pipeline = Compose(support_pipeline)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        if self.process_classes is not None:
            process_labels = [self.cat2label[cls_name] for cls_name in self.process_classes]
        for i, img_info in enumerate(self.img_infos):
            labels = self.get_ann_info(i)['labels']
            if len(labels) == 0:
                continue
            # if min(img_info['width'], img_info['height']) >= min_size:
            if self.process_classes is not None:
                ann_info = self.get_ann_info(i)
                labels_set = set(ann_info['labels'])
                have_intersection = labels_set & process_labels
                if have_intersection and (self.process_type == 'retain'):
                    valid_inds.append(i)
                elif (not have_intersection) and (self.process_type == 'filter'):
                    valid_inds.append(i)
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.enable_ignore:
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
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
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.dataset.CLASSES
            mean_ap, all_results = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_test_img(idx)
            if self.pre_test:
                support_data, support_labels  = self.prepare_pre_test_img(idx)
                data['support_instance_list'] = support_data
                data['support_instance_labels'] = support_labels
                # data['support_instance_list'] = 1
                # data['support_instance_labels'] = 2
            return data
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            if self.instance_path is not None:
                labels = data['gt_labels'].data
                support_data, support_labels = self.prepare_support_instances(labels)
                data['img_metas'].data['support_instance_list'] = support_data
                data['img_metas'].data['support_instance_labels'] = support_labels
            return data

    def prepare_pre_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.load_img(results)
        results = self.load_anno(results)
        s_img = results['img']
        s_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        support_patches = mi.imcrop(s_img, s_bboxes)
        support_instances = []
        for i in range(len(support_patches)):
            result = dict(img=mmcv.imread(support_patches[i]))
            support_instances.append(self.support_pipeline(result)['img'])
        # support_labels = to_tensor(support_labels)
        # support_labels = to_tensor(s_labels)

        return support_instances, to_tensor(labels)

    def prepare_support_instances(self, labels):
        labels = np.unique(labels)
        if labels.shape[0] > self.num_support_instances:
            labels = np.random.choice(labels, self.num_support_instances, replace=False)
        elif labels.shape[0] < self.num_support_instances:
            candidates = np.setdiff1d(list(range(1, len(self.CLASSES)+1)), labels)
            extra_labels = np.random.choice(candidates, self.num_support_instances - labels.shape[0], replace=False)
            labels = np.concatenate([extra_labels, labels], axis=0)
        support_instances = []
        for l in labels:
            class_name = self.CLASSES[l-1]
            instance_file_list = self.instance_files_dict[class_name]
            instance_file = random.choice(instance_file_list)
            result = dict(img=mmcv.imread(instance_file))
            result = self.support_pipeline(result)
            support_instances.append(result)
        return support_instances, to_tensor(labels)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

@DATASETS.register_module
class FSMetaVOCDataset(MetaVOCDataset):
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
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
            if self.year == 2007:
                ds_name = 'voc07'
            else:
                ds_name = self.dataset.CLASSES
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
                print_log(('| novel_set' + str(i + 1)).ljust(14) + '| mAP : ' +
                          str(np.around(ns_mAP, 3)).ljust(5, '0') + ' |', logger=logger)
            print_log('+-------------+-----+-------+', logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results

@DATASETS.register_module
class MetaVOCDatasetBase1(MetaVOCDataset):
    CLASSES = ('aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable',
               'dog', 'horse', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor')

@DATASETS.register_module
class MetaVOCDatasetBase2(MetaVOCDataset):
    CLASSES = ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog',
               'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor')

@DATASETS.register_module
class MetaVOCDatasetBase3(MetaVOCDataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor')

from mmdet.core import eval_map, eval_recalls
from .registry import DATASETS
from .xml_style import XMLDataset
import numpy as np

from mmdet.utils import print_log


@DATASETS.register_module
class VOCCOCODataset(XMLDataset):
    CLASSES_COCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    NOVEL_CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                     'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                     'potted plant', 'sheep', 'couch', 'train', 'tv',)

    BASE_CLASSES = ('truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                    'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    CLASSES_VOC = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    def __init__(self, **kwargs):
        super(VOCCOCODataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

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
            # if self.year == 2007:
            #     ds_name = 'voc07'
            # else:
            #     ds_name = self.dataset.CLASSES
            ds_name = 'voc07_coco'
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

            ns_ap = [all_results[self.CLASSES.index(c)]['ap'] for c in self.CLASSES_VOC]
            ns_mAP = np.array(ns_ap).mean()
            print_log(('| voc_set').ljust(14) + '| mAP : ' +
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
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i,
                                                                            j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results

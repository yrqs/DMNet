from mmdet.core import eval_map, eval_recalls
from .registry import DATASETS
from .xml_style import XMLDataset
import numpy as np

from mmdet.utils import print_log


@DATASETS.register_module
class VOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
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
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i,
                                                                            j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results

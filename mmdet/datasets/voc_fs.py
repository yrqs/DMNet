from mmdet.core import eval_map, eval_recalls
from .registry import DATASETS
from .xml_style import XMLDataset
import numpy as np


@DATASETS.register_module
class VOCDatasetFs(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]
    # CLASSES = ('diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')
    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat')

    def __init__(self, **kwargs):
        # self.n_shot = 10
        super(VOCDatasetFs, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        # valid_inds = []
        # for i, img_info in enumerate(self.img_infos):
        #     labels = self.get_ann_info(i)['labels']
        #     # print(labels)
        #     if len(labels) == 0:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 21
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 21
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
        labels_num = [0]*21
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds

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
class VOCDataset10s(VOCDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 10
        super(VOCDataset10s, self).__init__(**kwargs)

@DATASETS.register_module
class VOCDataset5s(VOCDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 5
        super(VOCDataset5s, self).__init__(**kwargs)

@DATASETS.register_module
class VOCDataset3s(VOCDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 3
        super(VOCDataset3s, self).__init__(**kwargs)

@DATASETS.register_module
class VOCDataset2s(VOCDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 2
        super(VOCDataset2s, self).__init__(**kwargs)

@DATASETS.register_module
class VOCDataset1s(VOCDatasetFs):
    def __init__(self, **kwargs):
        self.n_shot = 1
        super(VOCDataset1s, self).__init__(**kwargs)


@DATASETS.register_module
class VOCDatasetNovel1_1s(VOCDatasetFs):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    def __init__(self, **kwargs):
        self.n_shot = 1
        super(VOCDatasetNovel1_1s, self).__init__(**kwargs)
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        # valid_inds = []
        # for i, img_info in enumerate(self.img_infos):
        #     labels = self.get_ann_info(i)['labels']
        #     # print(labels)
        #     if len(labels) == 0:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 21
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 21
            for l in labels:
                if self.CLASSES[l-1] not in self.novel_sets[0] or have_label_list[l] + temp_list[l] >= self.n_shot:
                    have_label_set = set()
                    break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0]*21
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds


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
                # dataset='voc07novel1',
                dataset=ds_name,
                logger=logger)
            # novel_sets_mAP = []
            for i, ns in enumerate([self.novel_sets[0]]):
                ns_ap = [all_results[self.CLASSES.index(c)]['ap'] for c in ns]
                ns_mAP = np.array(ns_ap).mean()
                print(('| novel_set' + str(i + 1)).ljust(14) + '| mAP : ' +
                      str(np.around(ns_mAP, 3)).ljust(5, '0') + ' |')
            print('+-------------+-----+-------+')

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


@DATASETS.register_module
class VOCDatasetBase(XMLDataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    def __init__(self, **kwargs):
        super(VOCDatasetBase, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        # valid_inds = []
        # for i, img_info in enumerate(self.img_infos):
        #     labels = self.get_ann_info(i)['labels']
        #     # print(labels)
        #     if len(labels) == 0:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        # labels_num = [0]*21
        # for i in valid_inds:
        #     ann_info = self.get_ann_info(i)
        #     labels = list(set(ann_info['labels']))
        #     for l in labels:
        #         labels_num[l] += 1
        # print('labels_num: ', labels_num)
        # # print('valid_inds: ', valid_inds)
        valid_inds = []
        have_label_list = [0] * 21
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 21
            for l in labels:
                if self.CLASSES[l - 1] in self.novel_sets[self.novel_index] or have_label_list[l] + temp_list[l] >= 1e10:
                    have_label_set = set()
                    break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0] * 21
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        return valid_inds


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
            # base_idx = []
            # for c in self.CLASSES:
            #     if c not in self.novel_sets[self.novel_index]:
            #         base_idx.append(self.CLASSES.index(c))
            # base_ap = [all_results[i]['ap'] for i in base_idx]
            # base_mAP = np.array(base_ap).mean()
            # print(('| base_set' + str(self.novel_index + 1)).ljust(14) + '| mAP : ' +
            #       str(np.around(base_mAP, 3)).ljust(5, '0') + ' |')
            # print('+-------------+-----+-------+')
            # for i, ns in enumerate([self.novel_sets[self.novel_index]]):
            #     ns_ap = [all_results[self.CLASSES.index(c)]['ap'] for c in ns]
            #     ns_mAP = np.array(ns_ap).mean()
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

@DATASETS.register_module
class VOCDatasetBase1(VOCDatasetBase):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        # valid_inds = []
        # for i, img_info in enumerate(self.img_infos):
        #     labels = self.get_ann_info(i)['labels']
        #     # print(labels)
        #     if len(labels) == 0:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 21
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 21
            for l in labels:
                if self.CLASSES[l-1] in self.novel_sets[0] or have_label_list[l] + temp_list[l] >= 1e10:
                    have_label_set = set()
                    break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0]*21
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds

@DATASETS.register_module
class VOCDatasetBase_1(VOCDatasetBase):
    CLASSES = ('aeroplane', 'bicycle', 'c1', 'boat', 'bottle', 'c2', 'car',
               'cat', 'chair', 'c3', 'diningtable', 'dog', 'horse',
               'c4', 'person', 'pottedplant', 'sheep', 'c5', 'train',
               'tvmonitor')
    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')

    novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
                  ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
                  ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        # valid_inds = []
        # for i, img_info in enumerate(self.img_infos):
        #     labels = self.get_ann_info(i)['labels']
        #     # print(labels)
        #     if len(labels) == 0:
        #         continue
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         valid_inds.append(i)
        valid_inds = []
        have_label_list = [0] * 21
        for i, img_info in enumerate(self.img_infos):
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            have_label_set = set()
            temp_list = [0] * 21
            for l in labels:
                if have_label_list[l] + temp_list[l] >= 5:
                    have_label_set = set()
                    break
                if min(img_info['width'], img_info['height']) >= min_size:
                    temp_list[l] += 1
                    have_label_set.add(l)
            if len(have_label_set) > 0:
                for idx, label_num in enumerate(temp_list):
                    have_label_list[idx] += label_num
                valid_inds.append(i)
        labels_num = [0]*21
        for i in valid_inds:
            ann_info = self.get_ann_info(i)
            labels = list(set(ann_info['labels']))
            for l in labels:
                labels_num[l] += 1
        print('labels_num: ', labels_num)
        # print('valid_inds: ', valid_inds)
        return valid_inds

@DATASETS.register_module
class VOCDatasetBase2(VOCDatasetBase):
    # CLASSES = ('n1', 'bicycle', 'bird', 'boat', 'n2', 'bus', 'car',
    #            'cat', 'chair', 'n3', 'diningtable', 'dog', 'n4',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'n5', 'train',
    #            'tvmonitor')
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    def __init__(self, **kwargs):
        self.novel_index = 1
        super(VOCDatasetBase2, self).__init__(**kwargs)

@DATASETS.register_module
class VOCDatasetBase3(VOCDatasetBase):
    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'n1', 'bottle', 'bus', 'car',
    #            'n2', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'n3', 'person', 'pottedplant', 'n4', 'n5', 'train',
    #            'tvmonitor')
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    def __init__(self, **kwargs):
        self.novel_index = 2
        super(VOCDatasetBase3, self).__init__(**kwargs)
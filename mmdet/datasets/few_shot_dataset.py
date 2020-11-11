import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS

@DATASETS.register_module
class FewShotDataset(CustomDataset):
    def __init__(self,
                 *args):
        super(FewShotDataset).__init__(*args)
        self.label2ids = {}
        self.id2labels = {}
        self.label_id_init()
        self.id_label_init()

    def label_id_init(self):
        for img_info in self.img_infos:
            img_id = img_info['id']
            ann_info = self.get_ann_info(img_id)
            labels = ann_info['labels']
            labels = list(set(labels))
            for l in labels:
                if l not in self.label2ids.keys():
                    self.label2ids[l] = []
                self.label2ids[l].append(img_id)

    def id_label_init(self):
        for img_info in self.img_infos:
            img_id = img_info['id']
            ann_info = self.get_ann_info(img_id)
            labels = ann_info['labels']
            labels = list(set(labels))
            if img_id not in self.id2labels.keys():
                self.id2labels[img_id] = []
            self.id2labels[img_id].append(labels)

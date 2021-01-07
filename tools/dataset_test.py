from mmdet.datasets.coco import COCO

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

if __name__ == '__main__':
    coco_dataset = COCO(data_root + 'annotations/instances_train2014_' + str(10) + 'shot_novel_standard.json')

from mmdet.datasets import CocoDataset
import mmcv

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# ann_file = data_root + 'annotations/instances_train2014_base.json'
ann_file = data_root + 'annotations/instances_train2014_10shot_novel_standard.json'

img_prefix=data_root + 'images/trainval2014/'

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
coco = CocoDataset(fixed_cls_idx=True, ann_file=ann_file, pipeline=pipeline, img_prefix=img_prefix)


# for key in coco.cat2label.keys():
#     print(str(key) + ' : ' + str(coco.cat2label[key]))

# print(len(coco.cat2label))
#
# print(coco.cat_ids)
#
# print(coco.coco.cats)

results = mmcv.load('test_out.pkl')
# print(results)
# print(coco.coco.cats)
# coco.evaluate(results)

# for result in results:
#     for cls_idx, result_per_cls in enumerate(result):
#         if result_per_cls.shape != (0, 5):
#             if cls_idx > 59:
#                 print(cls_idx)


# model settings
model = dict(
    type='RetinaNet',
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    # type='RetinaNet',
    #     pretrained='open-mmlab://resnet50_caffe',
    #     backbone=dict(
    #         type='ResNet',
    #         depth=50,
    #         num_stages=4,
    #         out_indices=(0, 1, 2, 3),
    #         frozen_stages=1,
    #         norm_cfg=dict(type='BN', requires_grad=False),
    #         norm_eval=True,
    #         style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='GADMLRPNHead',
        # type='GARetinaHead',
        # num_classes=81,
        num_classes=21,
        in_channels=256,
        # stacked_convs=4,
        emb_sizes = (1024, 128),
        num_modes = 5,
        sigma=0.5,
        finetuning=True,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        octave_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[1.0, 1.0, 1.0, 1.0],
        target_means=(.0, .0, .0, .0),
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=False,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0),
        loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    ga_assigner=dict(
        type='ApproxMaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.4,
        ignore_iof_thr=-1),
    ga_sampler=dict(
        type='RandomSampler',
        num=256,
        pos_fraction=0.5,
        neg_pos_ub=-1,
        add_gt_as_proposals=False),
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.0,
        ignore_iof_thr=-1),
    sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
    allowed_border=-1,
    pos_weight=-1,
    center_ratio=0.2,
    ignore_ratio=0.5,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'VOC2UW20'
data_root = 'data/underwater/'
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Resize', img_scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
        ann_file=data_root + 'train/annotations/train.json',
        img_prefix=data_root + 'train/image/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file='data/train/annotations/testA.json',
        # img_prefix=data_root + 'test-A-image/',
        ann_file=data_root+'test18ori/annotations/test18ori.json',
        img_prefix=data_root+'test18ori/image/',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='NoneOptimizer', lr=1e-5, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='NoneOptimizer')
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = None
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[7, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 5
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ga_dml_x101_32x4d_fpn_1x'
# load_from = 'work_dirs/ga_dml_rpn_x101_uw20_embsize64_alpha04_lr0005-7-10-12_CE_ind1/epoch_11.pth'
# resume_from = 'work_dirs/ga_dml_rpn_x101_uw20_embsize64_alpha04_lr0005-7-10-12_CE_ind1/epoch_11.pth'

load_from = 'work_dirs/ga_dml_rpn_x101_voc_embsize128_alpha015_lr0005-7-13-17_CE_ind1_beta004/epoch_16.pth'
resume_from = None
workflow = [('train', 1)]

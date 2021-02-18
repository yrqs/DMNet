# model settings

save_outs = False
shot = 3
shot_idx = [1, 2, 3, 5, 10].index(shot)
train_repeat_times = [30, 25, 20, 15, 10][shot_idx]
freeze = False
freeze1 = False

stacked_convs = 2

model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaDRTHead',
        num_classes=21,
        in_channels=256,
        stacked_convs=stacked_convs,
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
        loc_filter_thr=0.1,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0)))
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
    allowed_border=-1,
    pos_weight=-1,
    center_ratio=0.2,
    ignore_ratio=0.5,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='soft_nms', iou_thr=0.3, min_score=0.0001),
    max_per_img=100)
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand'),
    dict(type='MinIoURandomCrop'),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
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
        img_scale=(1000, 600),
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=train_repeat_times,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval_' + str(shot) + 'shot_novel_standard.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval_' + str(shot) + 'shot_novel_standard.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=[
        #     data_root + 'VOC2007/ImageSets/Main/trainval_1shot_novel_standard.txt',
        #     data_root + 'VOC2012/ImageSets/Main/trainval_1shot_novel_standard.txt'
        # ],
        # img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

evaluation = dict(interval=2, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.00025*2*1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[12, 16])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 18
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ga_retinanet_x101_32x4d_fpn_1x'
load_from = 'work_dirs/retina_drt_fpn_voc_base1_r1_lr00125x2x2_8_11_12_ind1_1/epoch_12.pth'
# load_from = 'work_dirs/retina_drt_s4_fpn_voc_base1_r1_lr00125x2x2_8_11_12_ind1_1/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]

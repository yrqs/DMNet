warmup_iters = 500
lr_step = [8, 11, 12]
interval = 12
lr_base = 0.00125
imgs_per_gpu = 2
gpu_num = 8

split_num = 2

VOC_base_ids = (
    (0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19),
    (1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19),
    (0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19),
)

VOC_novel_ids = (
    (2, 5, 9, 13, 17),
    (0, 4, 9, 12, 17),
    (3, 7, 13, 16, 17)
)

# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet101',
    freeze_backbone=False,
    freeze_rpn=False,
    freeze_shared_head=False,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=2,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch'),
    shared_head=dict(
        type='ResLayer',
        depth=101,
        stage=3,
        stride=2,
        dilation=1,
        style='pytorch',
        norm_cfg=norm_cfg,
        norm_eval=True),
    rpn_head=dict(
        type='RPNHead',
        grad_scale=0.0,
        in_channels=1024,
        feat_channels=1024,
        anchor_scales=[2, 4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2, aligned=True),
        out_channels=1024,
        featmap_strides=[16]),
    bbox_head=dict(
        type='FSCosBBoxHead',
        triplet_margin=0.05,
        cos_scale=8,
        grad_scale=0.75,
        with_avg_pool=True,
        roi_feat_size=7,
        in_channels=2048,
        num_classes=16,
        base_ids=VOC_base_ids[split_num-1],
        novel_ids=VOC_novel_ids[split_num-1],
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=True,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=True,
        nms_pre=6000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
dataset_type = 'VOCDatasetBase{}'.format(split_num)
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[
        (1000, 302), (1000, 334), (1000, 376), (1000, 408),
        (1000, 440), (1000, 472), (1000, 504), (1000, 536),
        (1000, 568), (1000, 600)], multiscale_mode='value', keep_ratio=True),
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
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                # data_root + 'VOC2007/ImageSets/Main/trainval_split' + str(split_num) + '_base.txt',
                # data_root + 'VOC2012/ImageSets/Main/trainval_split' + str(split_num) + '_base.txt'
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test_split' + str(split_num) + '_base.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test_split' + str(split_num) + '_base.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

evaluation = dict(interval=2, metric='mAP')

# optimizer
optimizer = dict(type='SGD', lr=lr_base*imgs_per_gpu*gpu_num, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=1.0 / 3,
    step=[lr_step[0], lr_step[1]])
checkpoint_config = dict(interval=interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = lr_step[2]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_caffe_c4_1x'
load_from = None
resume_from = None
resume_optimizer = False
workflow = [('train', 1)]

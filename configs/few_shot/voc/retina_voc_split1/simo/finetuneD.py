shot = 1
shot_idx = [1, 2, 3, 5, 10]
train_repeat_times = [30, 25, 20, 15, 10]

stacked_convs = 0
warmup_iters = 500 // 8
lr_step = [8, 11, 12]
interval = 4
lr_base = 0.00025
imgs_per_gpu = 4
gpu_num = 8

split_num = 1

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
    type='RetinaNet',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=2,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='SIMO',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=3,
        num_outs=5,
        num_down=2),
    bbox_head=dict(
        type='RetinaHead',
        stacked_convs=0,
        grad_scale=0.001,
        num_classes=21,
        in_channels=256,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[
        (1333, 480), (1333, 512), (1333, 544), (1333, 576), (1333, 608),
        (1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
        (1333, 800)], multiscale_mode='value', keep_ratio=True),
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
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval_' + 'n' + 'shot_novel_standard.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval_' + 'n' + 'shot_novel_standard.txt'
                # data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                # data_root + 'VOC2012/ImageSets/Main/trainval.txt'
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
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
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
work_dir = './work_dirs/ga_dml_x101_32x4d_fpn_1x'
load_from = 'work_dirs/retina_voc_split1/simo/base/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]

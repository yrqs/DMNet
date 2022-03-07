import os

# model settings

save_outs = False
shot = 2
shot_idx = [1, 2, 3, 5, 10].index(shot)
train_repeat_times = [30, 25, 20, 15, 10][shot_idx]
freeze = False
freeze1 = False
neg_pos_ratio = 3
emb_sizes = [(256, 64), (256, 128), (512, 64), (256, 32),
             (512, 128), (256, 256), (128, 128), (128, 64),
             (128, 256)][1]
stacked_convs = 2

alpha = 0.15

warmup_iters = 500
lr_step = [12, 16, 18]
interval = 2
lr_base = 0.0001
imgs_per_gpu = 2
gpu_num = 2

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
        type='RetinaDMLHead3',
        num_classes=21,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        cls_emb_head_cfg=dict(
            emb_channels=(256, 128),
            num_modes=1,
            sigma=0.5,
            cls_norm=False),
        octave_base_scale=4,
        scales_per_octave=1,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0)))
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
    imgs_per_gpu=imgs_per_gpu,
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

evaluation = dict(interval=interval, metric='mAP')

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
load_from = 'work_dirs/retina_dml3_voc_split1/base_aug/wo_norm/base/epoch_16.pth'
resume_from = None
workflow = [('train', 1)]

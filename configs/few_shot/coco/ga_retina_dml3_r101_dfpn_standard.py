# model settings

save_outs = False
shot = 10
shot_idx = [10, 30].index(shot)
train_repeat_times = [10, 3][shot_idx]
freezeB = False
freeze = False
freeze1 = False
neg_pos_ratio = 10

model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4 if freezeB else 1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='DFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        save_outs=save_outs,
        freeze=freeze),
    bbox_head=dict(
        type='GARetinaDMLHead3D',
        num_classes=81,
        in_channels=256,
        stacked_convs=2,
        emb_sizes = (256, 64),
        num_modes = 1,
        sigma=0.5,
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
        freeze=freeze1,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.7),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0),
        loss_emb=dict(type='RepMetLoss', alpha=0.10, loss_weight=1.0)))
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
        min_pos_iou=0,
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
    neg_pos_ratio=neg_pos_ratio,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='soft_nms', iou_thr=0.3, min_score=0.0001),
    # nms=dict(type='nms', iou_thr=0.3),
    max_per_img=100)
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Expand'),
    # dict(type='MinIoURandomCrop'),
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
            fixed_cls_idx=True,
            ann_file=[data_root + 'annotations/instances_train2014_' + str(shot) + 'shot_novel_standard.json',
                      data_root + 'annotations/instances_val2014_' + str(shot) + 'shot_novel_standard.json',
                      ],
            img_prefix=[data_root + 'images/trainval2014/', data_root + 'images/trainval2014/',],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        fixed_cls_idx=True,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/trainval2014/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        fixed_cls_idx=True,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/trainval2014/',
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.00025*2*2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10, 14])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 16
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ga_dml_x101_32x4d_fpn_1x'
load_from = 'work_dirs/ga_retina_dml3_dfpn_emb256_64_alpha010_le10_CE_nratio10_coco_standardr1_lr0001x2x2_10_14_16/epoch_16.pth'
resume_from = None
workflow = [('train', 1)]

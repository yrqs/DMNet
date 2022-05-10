import torch

# checkpoint_path = 'work_dirs/retina_x101_32x4d_fpn_1x_coco1333x800_base1_r1_lr00125x2x1_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/gas_retina_x101_32x4d_dfpn_1x_coco1333x800_base1_r1_lr00125x2x2_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/faster_rcnn_x101_32x4d_fpn_1x_coco1000x600_base_r1_lr00125x2x2_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/frcn_r101_dfpnv3_1x_voc_base1_r1_lr00125x2x2_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/frcn_r101_fpn_1x_voc_base_r1_lr00125x2x4_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/retina_r101_fpn_voc_base1_r1_lr00125x2x1_8_11_12/epoch_12.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml_fpn_emb1024_256_alpha015_le025_coco_base1_r1_lr000125x2x4_10_14_16/epoch_12.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x1_10_14_16_ind2_1/epoch_16.pth'
# checkpoint_path = '/home/luyue/Documents/mmdetection_old/work_dirs/test/epoch_2.pth'
# checkpoint_path = '/home/luyue/.cache/torch/checkpoints/resnet101_caffe-3ad79236.pth'
# checkpoint_path = 'work_dirs/retina_dml3_coco/base_aug/norm/base/epoch_24.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml12_voc_split1/wo_norm/base/epoch_16.pth'
# checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/base/epoch_12.pth'
# checkpoint_path = '/home/luyue/.cache/torch/checkpoints/resnet101-5d3b4d8f_wo_res5.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/pre_base/base/epoch_16.pth'
checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/aug2/base/epoch_12.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

for key in checkpoint['state_dict'].keys():
    print(key)
# for key in checkpoint.keys():
#     print(key)

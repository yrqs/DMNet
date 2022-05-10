import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F

# checkpoint_path = 'work_dirs/ga_retina_dml3_dfpn2_emb256_64_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_10_14_16/epoch_16.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml9_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base2_r1_lr00025x2x2_10_14_16_ind1_1/epoch_16.pth'
checkpoint_path = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/default/base/epoch_16.pth'

checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

init_type = ['head', 'head1', 'head2', 'rep'][3]
changes = {
    'head':
        ['bbox_head.retina_cls.weight',
         'bbox_head.retina_cls.bias'],

    'head1':
        ['bbox_head.retina_cls.weight',
         'bbox_head.retina_cls.bias',
         'bbox_head.retina_reg.weight',
         'bbox_head.retina_reg.bias'],

    # 'head2':
    #     ['bbox_head.retina_cls.weight',
    #      'bbox_head.retina_cls.bias',
    #      'bbox_head.retina_reg.weight',
    #      'bbox_head.retina_reg.bias'],

    'rep':
        ['bbox_head.cls_head.rep_fc.weight',
         'bbox_head.cls_head.rep_fc.bias']
}[init_type]

def init_weights():
    for c in changes:
        if 'weight' in c:
            nn.init.normal_(checkpoint['state_dict'][c], mean=0, std=0.01)
        if 'bias' in c:
            if 'cls' in c and 'retina' in c:
                bias_cls = bias_init_with_prob(0.01)
                nn.init.constant_(checkpoint['state_dict'][c], bias_cls)
            else:
                nn.init.constant_(checkpoint['state_dict'][c], 0)

    new_checkpoint_path = checkpoint_path[:-4] + '_init_' + init_type + checkpoint_path[-4:]

    torch.save(checkpoint, new_checkpoint_path)

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

VOC_novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
              ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
              ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

VOC_novel_ids = (
    (2, 5, 9, 13, 17),
    (0, 4, 9, 12, 17),
    (3, 7, 13, 16, 17)
)

CLASSES = [VOC_CLASSES][0]
novel_sets = [VOC_novel_sets][0]
novel_index = 0

def change_rep():
    finetune_checkpoint_path = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/default/base/epoch_16.pth'
    finetune_checkpoint = checkpoint = torch.load(finetune_checkpoint_path, map_location=torch.device("cpu"))
    new_rep = finetune_checkpoint['state_dict']['bbox_head.representations']
    old_rep = checkpoint['state_dict']['bbox_head.representations']
    for n in novel_sets[novel_index]:
        cls_idx = CLASSES.index(n)
        old_rep[cls_idx] = new_rep[cls_idx]
    checkpoint['state_dict']['bbox_head.representations'] = old_rep
    new_checkpoint_path = checkpoint_path[:-4] + '_rep' + checkpoint_path[-4:]
    torch.save(checkpoint, new_checkpoint_path)

def init_novel_rep(novel_ids):
    rep_fc_w = checkpoint['state_dict']['bbox_head.cls_head.rep_fc.weight'].reshape(-1, 128)
    init_w = rep_fc_w.clone()
    nn.init.normal_(init_w, std=0.01)
    rep_fc_w[novel_ids, :] = init_w[novel_ids, :]

    rep_fc_b = checkpoint['state_dict']['bbox_head.cls_head.rep_fc.bias'].reshape(-1, 128)
    init_b = rep_fc_b.clone()
    nn.init.constant_(init_b, 0)
    rep_fc_b[novel_ids, :] = init_b[novel_ids, :]

    checkpoint['state_dict']['bbox_head.cls_head.rep_fc.weight'].data = rep_fc_w.reshape(-1, 1)
    checkpoint['state_dict']['bbox_head.cls_head.rep_fc.bias'].data = rep_fc_b.reshape(-1)

    new_checkpoint_path = checkpoint_path[:-4] + '_init_nrep' + checkpoint_path[-4:]
    torch.save(checkpoint, new_checkpoint_path)

def remove_res5():
    checkpoint_path = '/home/luyue/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'
    state_dict = torch.load(checkpoint_path)
    remove_keys = 'layer4.0.conv1.weight, layer4.0.bn1.running_mean, layer4.0.bn1.running_var, layer4.0.bn1.weight, layer4.0.bn1.bias, layer4.0.conv2.weight, layer4.0.bn2.running_mean, layer4.0.bn2.running_var, layer4.0.bn2.weight, layer4.0.bn2.bias, layer4.0.conv3.weight, layer4.0.bn3.running_mean, layer4.0.bn3.running_var, layer4.0.bn3.weight, layer4.0.bn3.bias, layer4.0.downsample.0.weight, layer4.0.downsample.1.running_mean, layer4.0.downsample.1.running_var, layer4.0.downsample.1.weight, layer4.0.downsample.1.bias, layer4.1.conv1.weight, layer4.1.bn1.running_mean, layer4.1.bn1.running_var, layer4.1.bn1.weight, layer4.1.bn1.bias, layer4.1.conv2.weight, layer4.1.bn2.running_mean, layer4.1.bn2.running_var, layer4.1.bn2.weight, layer4.1.bn2.bias, layer4.1.conv3.weight, layer4.1.bn3.running_mean, layer4.1.bn3.running_var, layer4.1.bn3.weight, layer4.1.bn3.bias, layer4.2.conv1.weight, layer4.2.bn1.running_mean, layer4.2.bn1.running_var, layer4.2.bn1.weight, layer4.2.bn1.bias, layer4.2.conv2.weight, layer4.2.bn2.running_mean, layer4.2.bn2.running_var, layer4.2.bn2.weight, layer4.2.bn2.bias, layer4.2.conv3.weight, layer4.2.bn3.running_mean, layer4.2.bn3.running_var, layer4.2.bn3.weight, layer4.2.bn3.bias, fc.weight, fc.bias'
    remove_keys = remove_keys.split(',')
    remove_keys = [k.strip() for k in remove_keys]
    for k in remove_keys:
        state_dict.pop(k)
    new_checkpoint_path = checkpoint_path[:-4] + 'wo_res5' + checkpoint_path[-4:]
    torch.save(state_dict, new_checkpoint_path)

def channel_pruning():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    model = torch.load(checkpoint_path)
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight'].clone()
    w_s = fc_cls_w.abs().sort(dim=1, descending=True)[0]
    thresh = w_s[:, 256].reshape(-1, 1).expand_as(fc_cls_w)
    print((fc_cls_w.abs() < thresh).sum())
    fc_cls_w[fc_cls_w.abs() < thresh] = 0.
    fc_cls_w[0, :] = model['state_dict']['bbox_head.fc_cls.weight'][0, :]
    model['state_dict']['bbox_head.fc_cls.weight'] = fc_cls_w
    new_checkpoint_path = checkpoint_path[:-4] + '_mask' + checkpoint_path[-4:]
    torch.save(model, new_checkpoint_path)

def channel_mask():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    model = torch.load(checkpoint_path)
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight'].clone()
    w_s = fc_cls_w.abs().sort(dim=1, descending=True)[0]
    thresh = w_s[:, 256].reshape(-1, 1).expand_as(fc_cls_w)
    print((fc_cls_w.abs() < thresh).sum())
    mask = torch.ones_like(fc_cls_w)
    mask[fc_cls_w.abs() < thresh] = 0.
    mask[0, :] = 1.

    cls_ids = (0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19)
    feat_list = torch.split(mask, 128, 1)
    for feat in feat_list:
        class_names = CLASSES
        if cls_ids is not None:
            feat = feat[[i+1 for i in cls_ids], :]
            class_names = [class_names[i] for i in cls_ids]
        num_cls = feat.shape[0]
        scale_ls = range(num_cls)
        label_ls = list(class_names)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.yticks(scale_ls, label_ls)
        plt.show()
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/base/epoch_12.pth'
    # model = torch.load(checkpoint_path)
    # model['state_dict']['bbox_head.cls_fc_w_mask'] = mask
    # new_checkpoint_path = checkpoint_path[:-4] + '_mask' + checkpoint_path[-4:]
    # torch.save(model, new_checkpoint_path)

if __name__ == '__main__':
    # init_novel_rep(VOC_novel_ids[0])
    # init_weights()
    # change_rep()
    # remove_res5()
    # channel_pruning()
    channel_mask()
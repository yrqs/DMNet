import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob

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

if __name__ == '__main__':
    init_novel_rep(VOC_novel_ids[0])
    # init_weights()
    # change_rep()
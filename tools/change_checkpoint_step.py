import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob

# checkpoint_path = 'work_dirs/ga_retina_dml6_s2_fpn_emb256_64_alpha015_le10_CE_nratio10_voc_aug_standard1_3shot_r20_lr0001x2x1_warm1000_10_18_20_ind1_1_1/epoch_18.pth'
# checkpoint_path = 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio10_voc_aug_standard1_3shot_r20_lr0001x2x1_warm1000_14_18_20_ind1_1_1/epoch_14.pth'
checkpoint_path = 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_aug_base1_r1_lr00025x2x2_10_14_16_ind2_1/epoch_10.pth'

checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

replace_dict = {
    'step=[10, 14]': 'step=[14, 18]',
    'total_epochs = 16': 'total_epochs = 20',
}

# replace_dict = {
#     'lr_step = [10, 14, 16]': 'lr_step = [14, 18, 20]',
# }

output_name = checkpoint_path[:checkpoint_path.rindex('/')] + '/' + 'epoch_10_new_step.pth'

if __name__ == '__main__':
    config = checkpoint['meta']['config']
    config_new = config
    for val_old, val_new in replace_dict.items():
        config_new.replace(val_old, val_new)
    checkpoint['meta']['config'] = config_new
    torch.save(checkpoint, output_name)

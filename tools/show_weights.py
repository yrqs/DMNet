import torch
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F


CLASSES_VOC = ('plane', 'bike', 'bird*', 'boat', 'bottle', 'bus*', 'car', 'cat',
               'chair', 'cow*', 'table', 'dog', 'horse', 'motorbike*', 'person',
               'pottedplant', 'sheep', 'sofa*', 'train', 'tv')

CLASSES = CLASSES_VOC

novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
              ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
              ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

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

# checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/base/epoch_12.pth'
checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
model = torch.load(checkpoint_path, map_location=torch.device("cpu"))['state_dict']

cls_ids = VOC_base_ids[0]

def show_fc_cls_weight(weight):
    feat_list = torch.split(weight, 128, 1)
    for feat in feat_list:
        class_names = CLASSES
        if cls_ids is not None:
            feat = feat[cls_ids, :]
            class_names = [class_names[i] for i in cls_ids]
        num_cls = feat.shape[0]
        scale_ls = range(num_cls)
        label_ls = list(class_names)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.yticks(scale_ls, label_ls)
        plt.show()

def show_fc_reg_weight(weight):
    feat_list = torch.split(weight, 128, 1)
    for feat in feat_list:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.show()

def show_fc_weight(weight, class_names=None, norm=(-0.25, 0.25)):
    norm_min, norm_max = norm
    feat_list = torch.split(weight, 128, 1)
    for i, feat in enumerate(feat_list):
        norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        if class_names is not None:
            num_cls = len(class_names)
            scale_ls = range(num_cls)
            label_ls = list(class_names)
            plt.yticks(scale_ls, label_ls)
        plt.title('%d - %d' % (i*128, (i+1)*128-1))
        plt.show()

def show_fc_cls_reg_weight(cls_ids=None):
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/cls_w_l1_regular/base/epoch_12.pth'
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/cls_w_l1_regular/D/1shot/epoch_16.pth'
    checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/DF/10shot/epoch_16.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight']
    fc_reg_w = state_dict['bbox_head.fc_reg.weight']
    bg_w = fc_cls_w[0, :].reshape(1, -1)
    fg_w = fc_cls_w[1:, :]
    fg_w = F.normalize(fg_w, dim=1, p=2)
    cls_names = list(CLASSES)
    if cls_ids is not None:
        fg_w = fg_w[cls_ids, :]
        cls_names = [cls_names[i] for i in cls_ids]
    cls_names = ['bg'] + cls_names + ['x', 'y', 'w', 'h']
    feat = torch.cat([bg_w, fg_w, fc_reg_w], dim=0)
    show_fc_weight(feat, cls_names, norm=(-0.25, 0.25))

def show_fc_cls_reg_weight_cossim(cls_ids=None):
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight']
    fc_reg_w = state_dict['bbox_head.fc_reg.weight']
    bg_w = fc_cls_w[0, :].reshape(1, -1)
    fg_w = fc_cls_w[1:, :]
    cls_names = list(CLASSES)
    if cls_ids is not None:
        fg_w = fg_w[cls_ids, :]
        cls_names = [cls_names[i] for i in cls_ids]
    cls_names = ['bg'] + cls_names + ['x', 'y', 'w', 'h']
    feat = torch.cat([bg_w, fg_w, fc_reg_w], dim=0)

    feat_norm = F.normalize(feat, dim=1, p=2)
    cos_sim = (feat_norm[None, :, :].expand(feat.shape[0], -1, -1) * feat_norm[:, None, :].expand(-1, feat.shape[0], -1)).sum(-1)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt.imshow(cos_sim, cmap='rainbow', norm=norm)
    plt.colorbar()
    num_cls = len(cls_names)
    scale_ls = range(num_cls)
    label_ls = list(cls_names)
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    plt.show()

def show_fc_cls_reg_weight_l2dis(cls_ids=None):
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight']
    fc_reg_w = state_dict['bbox_head.fc_reg.weight']
    bg_w = fc_cls_w[0, :].reshape(1, -1)
    fg_w = fc_cls_w[1:, :]
    cls_names = list(CLASSES)
    if cls_ids is not None:
        fg_w = fg_w[cls_ids, :]
        cls_names = [cls_names[i] for i in cls_ids]
    cls_names = ['bg'] + cls_names + ['x', 'y', 'w', 'h']
    feat = torch.cat([bg_w, fg_w, fc_reg_w], dim=0)

    feat_norm = F.normalize(feat, dim=1, p=2)
    # feat_norm = feat
    ex_size = torch.Size((feat.size(0), feat.size(0), feat.size(1)))
    l2_dis = ((feat_norm[None, :, :].expand(ex_size) - feat_norm[:, None, :].expand(ex_size))**2).sum(-1)
    l2_dis = torch.sqrt(l2_dis)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2.5)
    plt.imshow(l2_dis, cmap='rainbow', norm=norm)
    plt.colorbar()
    num_cls = len(cls_names)
    scale_ls = range(num_cls)
    label_ls = list(cls_names)
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    plt.show()

def show_bn_w():
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    checkpoint_path = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/pre_base/base/epoch_16.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    # bn_w = state_dict['shared_head.layer4.2.bn3.weight'].reshape(1, -1)
    bn_w = state_dict['bbox_head.cls_head.emb_module.1.weight'].reshape(1, -1)
    show_fc_weight(bn_w, norm=(0, 1))

def show_spicial_att_w():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/aug2/base/epoch_12.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    spicial_att_w = state_dict['bbox_head.spicial_att.weight'].reshape(1, -1)
    show_fc_weight(spicial_att_w, norm=(-0.1, 0.1))

def show_channel_mask():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/1shot/epoch_16.pth'
    model = torch.load(checkpoint_path)
    state_dict = model['state_dict']
    fc_cls_w = state_dict['bbox_head.fc_cls.weight'].clone()
    w_s = fc_cls_w.abs().sort(dim=1, descending=True)[0]
    thresh = w_s[:, 256].reshape(-1, 1).expand_as(fc_cls_w)
    print((fc_cls_w.abs() < thresh).sum())
    mask_cls = torch.ones_like(fc_cls_w)
    mask_cls[fc_cls_w.abs() < thresh] = 0.
    mask_cls[0, :] = 1.

    fc_reg_w = state_dict['bbox_head.fc_reg.weight'].clone()
    w_s = fc_reg_w.abs().sort(dim=1, descending=True)[0]
    thresh = w_s[:, 256].reshape(-1, 1).expand_as(fc_reg_w)
    print((fc_reg_w.abs() < thresh).sum())
    mask_reg = torch.ones_like(fc_reg_w)
    mask_reg[fc_reg_w.abs() < thresh] = 0.

    cls_ids = (0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19)
    mask_cat = torch.cat([mask_cls, mask_reg], dim=0)
    feat_list = torch.split(mask_cat, 128, 1)
    for feat in feat_list:
        class_names = CLASSES + ('x', 'y', 'w', 'h')
        if cls_ids is not None:
            feat = feat[[i+1 for i in cls_ids] + [-4, -3, -2, -1], :]
            class_names = [class_names[i] for i in (list(cls_ids) + [-4, -3, -2, -1])]
        num_cls = feat.shape[0]
        scale_ls = range(num_cls)
        label_ls = list(class_names)
        print(scale_ls, label_ls)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.yticks(scale_ls, label_ls)
        plt.show()

def show_channel_attention():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/mutex_channel_attention/base/epoch_12.pth'
    model = torch.load(checkpoint_path)
    state_dict = model['state_dict']
    cls_att = state_dict['bbox_head.mutex_channel_attention.channel_attention']
    reg_att = F.relu(torch.ones_like(cls_att) - cls_att)
    cat_att = torch.cat([cls_att[None, :], reg_att[None, :]])
    feat_list = torch.split(cat_att, 128, 1)
    for feat in feat_list:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.show()

def show_l1_channel_attention():
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/l1_channel_attention/base/epoch_12.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    for k in state_dict.keys():
        print(k)
    cls_att = state_dict['bbox_head.cls_channel_attention.channel_attention']
    reg_att = state_dict['bbox_head.reg_channel_attention.channel_attention']
    cat_att = torch.cat([cls_att[None, :], reg_att[None, :]])
    feat_list = torch.split(cat_att, 128, 1)
    for feat in feat_list:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.show()

def show_use_cos_sim(cls_ids=None):
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/DF/10shot/epoch_16.pth'
    checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/aug2/DF/10shot/epoch_16.pth'
    # checkpoint_path = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/DF/10shot/epoch_16.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    cls_w = state_dict['bbox_head.fc_cls.weight']

    cls_names = list(CLASSES)
    if cls_ids is not None:
        cls_w = cls_w[[i+1 for i in cls_ids], :]
        cls_names = [cls_names[i] for i in cls_ids]
    cls_names = ['bg'] + cls_names

    cls_w_norm = F.normalize(cls_w, dim=1, p=2)
    cls_w_norm_ex1 = cls_w_norm[None, :, :].expand(cls_w_norm.size(0), -1, -1)
    cls_w_norm_ex2 = cls_w_norm[:, None, :].expand_as(cls_w_norm_ex1)
    cos_sim = F.cosine_similarity(cls_w_norm_ex1, cls_w_norm_ex2, dim=2)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    plt.imshow(cos_sim, cmap='rainbow', norm=norm)
    plt.colorbar()
    num_cls = len(cls_names)
    scale_ls = range(num_cls)
    label_ls = list(cls_names)
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    plt.xticks(rotation=270)
    plt.show()

def show_fc_cls_w_max():
    checkpoint_path = 'work_dirs/frcn_r101_voc/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/split2/base/epoch_12.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['state_dict']
    cls_w = state_dict['bbox_head.fc_cls.weight']
    cls_w[1:, :] = F.normalize(cls_w[1:, :], p=2, dim=1)
    cls_w_max = cls_w.max(1)[0][None, :]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=cls_w_max.max())
    plt.imshow(cls_w_max, cmap='rainbow', norm=norm)
    plt.show()

if __name__ == '__main__':
    # fc_cls_w = model['bbox_head.fc_cls.weight']
    # fc_reg_w = model['bbox_head.fc_reg.weight']
    # fc_cls_b = model['bbox_head.fc_cls.bias']
    # print(fc_cls_b)
    # show_fc_cls_weight(fc_cls_w.abs())
    # show_fc_reg_weight(fc_reg_w.abs())
    # show_channel_attention()
    # show_channel_mask()
    # show_l1_channel_attention()
    # show_fc_cls_reg_weight(VOC_base_ids[0])
    # show_fc_cls_reg_weight()
    # show_bn_w()
    # show_fc_cls_reg_weight_cossim(VOC_base_ids[0])
    # show_fc_cls_reg_weight_cossim()
    # show_fc_cls_reg_weight_l2dis()
    # show_spicial_att_w()
    # show_use_cos_sim()
    show_fc_cls_w_max()
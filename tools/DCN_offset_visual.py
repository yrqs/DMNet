from mmdet.datasets import build_dataloader, build_dataset
import torch
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import numpy as np
from mmcv import Config

config_file = 'configs/few_shot/voc/voc_test.py'

file_outs_dict  = {
    'ga_retina_dml3' : 'ga_retina_dml3_feature.pth',
    'ga_retina_dml7' : 'ga_retina_dml7_feature.pth',
    'ga_retina_dml8' : 'ga_retina_dml8_feature.pth',
    'ga_retina' : 'ga_retina_feature.pth',
    'ga_retina_dmlneg3' : 'ga_retina_dmlneg3_feature.pth',
    'ga_retina_dmlneg6' : 'ga_retina_dmlneg6_feature.pth',
    'ga_retina_dmlneg7' : 'ga_retina_dmlneg7_feature.pth',
}

offset_base = [[-1, -1], [0, -1], [1, -1],
               [-1,  0], [0,  0], [1,  0],
               [-1,  1], [0,  1], [1,  1],]

def show_offset_all(offsets, level, img_h):
    group_num = offsets_cls[0].size(1) // 9 // 2
    steps = []
    positions_ori_list = []
    positions_offset_list = []
    for offset in [offsets[level]]:
        offset = offset.squeeze(0)
        feature_w = offset.size(2)
        feature_h = offset.size(1)
        step = img_h // feature_h
        steps.append(step)
        centers = torch.ones_like(offset[0]).nonzero().float() + 0.5
        centers = centers.view(offset.size(1), offset.size(2), 2)
        coordinates_list = []
        # conv kernel original receptive field
        for o in offset_base:
            coordinates_list.append(centers + torch.tensor(o).float())
        coordinates = torch.cat(coordinates_list, dim=2)
        coordinates = coordinates.repeat(1, 1, group_num)
        # (offsets, w, h) -> (w, h, offsets)
        offset = offset.permute(1, 2, 0).contiguous()

        offset_cls_x = offset.view(-1, 2)[:, 0]
        offset_cls_y = offset.view(-1, 2)[:, 1]
        offset = torch.cat([offset_cls_y.unsqueeze(-1), offset_cls_x.unsqueeze(-1)], dim=-1).view_as(offset)

        coordinates_ori = coordinates
        coordinates_offset = coordinates_ori + offset
        coordinates_offset_x = coordinates_offset.view(-1, 2)[:, 1].clamp(0.5, feature_w)
        coordinates_offset_y = coordinates_offset.view(-1, 2)[:, 0].clamp(0.5, feature_h)
        coordinates_offset = torch.cat([coordinates_offset_x.unsqueeze(-1), coordinates_offset_y.unsqueeze(-1)], dim=-1).view_as(coordinates_offset)
        print(coordinates_offset.size())
        # resize to origin image
        positions_ori = (coordinates_ori * step).ceil()
        positions_ori_list.append(positions_ori)
        positions_offset = (coordinates_offset * step).ceil()
        positions_offset_list.append(positions_offset)

    plt.imshow(img)
    x = 2
    y = 2
    for positions_ori, positions_offset in zip(positions_ori_list, positions_offset_list):
        pos_ori = positions_ori[y, x]
        # pos_ori = positions_ori[:, :]
        pos_ori = pos_ori.view(-1, 2)
        pos_offset = positions_offset[y, x]
        # pos_offset = positions_offset[:, :]
        pos_offset = pos_offset.view(-1, 2)
        plt.scatter(pos_ori[:, 1], pos_ori[:, 0], s=5, c='y', alpha=1)
        plt.scatter(pos_offset[:, 0], pos_offset[:, 1], s=5, c='r', alpha=1)
    plt.show()

def show_offset_center(offsets, step, level, group=0):
    offset = offsets[level]
    offset = offset.squeeze(0)
    centers = torch.ones_like(offset[0]).nonzero().float()+0.5
    centers[:, 1] += 1
    # print(centers.size())
    centers = centers.view(offset.size(1), offset.size(2), 2)

    centers = centers + torch.tensor(offset_base[1]).float()

    feature_w = offset.size(2)
    feature_h = offset.size(1)
    offset = offset.permute(1, 2, 0).contiguous()
    print(offset.size())
    centers_offsets = []
    x_idx = 9 + 18*group
    y_idx = 8 + 18*group

    scale = 10

    # offset_x = offset[:, :, x_idx].unsqueeze(-1) * feature_w
    # offset_y = offset[:, :, y_idx].unsqueeze(-1) * feature_h
    offset_x = offset[:, :, x_idx].unsqueeze(-1)
    offset_y = offset[:, :, y_idx].unsqueeze(-1)

    centers_offset = centers + torch.cat([offset_y, offset_x], dim=-1)

    coordinates_offset_x = centers_offset.view(-1, 2)[:, 1].clamp(-0.5, feature_w+1)
    coordinates_offset_y = centers_offset.view(-1, 2)[:, 0].clamp(-0.5, feature_h+1)
    centers_offset = torch.cat([coordinates_offset_y.unsqueeze(-1), coordinates_offset_x.unsqueeze(-1)], dim=-1).view_as(centers_offset)

    centers_offsets = centers_offset
    centers *= step
    centers_offsets *= step

    centers = centers.view(-1, 2)
    centers_offsets = centers_offsets.view(-1, 2)
    plt.figure(i)
    plt.imshow(img)
    plt.scatter(centers[:, 1], centers[:, 0], s=60, linewidths=0, marker='o', c='skyblue', alpha=0.7)
    plt.scatter(centers_offsets[:, 1], centers_offsets[:, 0], s=60, marker='o', c='r', alpha=1)
    plt.axis('off')
    plt.show()

show_list = [43, 44, 46, 47, ]
# show_list = [6, 8, 9, 17, 42, 43, 44, 46, 47, ]
# show_list = [13, 16, 18, 32, 34, 38, 43, 44, 46, 47, ]
not_show_list = []

filter_show = True
filter_not_show = False

stride = [8, 16, 32, 64, 128]

# group_list = (0, 1, 2, 3)
group_list = (0, )
# group_list = (2, )

if __name__ == '__main__':

    feature_type = 'ga_retina_dmlneg7'
    # feature_type = 'ga_retina_dmlneg3'
    # feature_type = 'ga_retina_dmlneg6'
    # root_path = 'mytest/'
    # root_path = 'mytest/ga_dmlneg3_base1_t3s/'
    # root_path = 'mytest/ga_dmlneg3_10s_t3s/'
    root_path = 'mytest/ga_dmlneg7_base2_t3s/'
    # root_path = 'mytest/ga_dmlneg3_base2_t3s/'
    # root_path = 'mytest/ga_dmlneg6_base2_t3s/'

    file_name_base = file_outs_dict[feature_type]
    file_name_base = root_path + file_name_base

    cfg = Config.fromfile(config_file)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    for i, data in enumerate(data_loader):
        if filter_show and i not in show_list:
            continue
        img = data['img'][0]
        img = img.squeeze(0)
        img = img.flip(dims=[2])
        img_h = img.shape[0]
        img_w = img.shape[1]

        file_name = file_name_base[:-4] + str(i+1) + file_name_base[-4:]
        outs = torch.load(file_name, map_location=torch.device("cpu"))

        offsets_cls = outs['offsets_cls']
        level = -2
        feature_h = offsets_cls[level].size(2)
        # step = img_h // feature_h
        # print(step)
        step = stride[level]
        for group in group_list:
            show_offset_center(offsets_cls, step, level, group)
        # show_offset_all(offsets_cls, level, img_h)
        # offsets_cls = outs['offsets_reg']
        # group_num = offsets_cls[0].size(1) // 9 // 2

        # offsets_reg = outs['offsets_reg']
        # show_offset_center(offsets_reg, step, level)

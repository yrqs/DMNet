import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F

CLASSES_VOC = ('plane', 'bike', 'bird*', 'boat', 'bottle', 'bus*', 'car', 'cat',
               'chair', 'cow*', 'table', 'dog', 'horse', 'motorbike*', 'person',
               'pottedplant', 'sheep', 'sofa*', 'train', 'tv')

CLASSES_COCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CLASSES = CLASSES_VOC
# CLASSES = CLASSES_COCO

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

# checkpoint_file = 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_10_14_16_ind1_1/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split2/wo_norm/256_256/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split2/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/retina_dml3_voc_split1/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/retina_dml3_voc_split1/base_aug/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/retina_dml3_voc_split1/norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/retina_dml3_voc_split1/base_aug/norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml12_voc_split1/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml9_voc_split2/wo_norm/10shot/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml9_voc_split2/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml11_voc_split2/wo_norm/256_256/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml6_voc_split2/wo_norm/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/default/1shot/epoch_4.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/default/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/sigma025_alpha03/1shot/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/default/10shot/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_voc_split1/wo_norm/pre_base/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml17_voc_split1/wo_norm/default/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml17_voc_split1/wo_norm/default/10shot/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml17_voc_split1/wo_norm/default/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml17_voc_split1/wo_norm/base_aug/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml16_voc_split1/norm/default/10shot/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml16_voc_split1_old/wo_norm/default/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml16_voc_split1/wo_norm/default/base/epoch_16.pth'
checkpoint_file = 'work_dirs/ga_retina_dml14_voc_split1/pre_base/base/epoch_16.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_coco/wo_norm/base/epoch_20.pth'
# checkpoint_file = 'work_dirs/ga_retina_dml4_coco/wo_norm/30shot/epoch_20.pth'

checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
# for key in checkpoint['state_dict'].keys():
#     print(key)
reps = checkpoint['state_dict']['bbox_head.cls_head.representations']

reps_fc_weight = checkpoint['state_dict']['bbox_head.cls_head.rep_fc.weight']
reps_fc_bias = checkpoint['state_dict']['bbox_head.cls_head.rep_fc.bias'].reshape_as(reps_fc_weight)
reps_ori = (reps_fc_weight + reps_fc_bias).reshape_as(reps)

if 'dml11' in checkpoint_file:
    reps = checkpoint['state_dict']['bbox_head.cls_head.representations'].expand_as(reps) + reps
    reps = reps.view(len(CLASSES), 1, -1)
    reps = F.normalize(reps, p=2, dim=2)

# dim = 2
dim = 2

shapes = ['o', 'v', '^', '<', '>', 's', '*', 'p', 'P', 'X', 'D', '$\dag$', 'H', '+', 'x', '|', '_', ]

colors = plt.cm.Spectral(range(len(CLASSES_VOC)))

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def reps_visual(reps, dim=2):
    '''t-SNE'''
    num_cls = reps.size(0)
    num_mode = reps.size(1)
    reps = reps.numpy().reshape(-1, reps.size(-1))
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    tsne = manifold.TSNE(n_components=dim, init='pca', perplexity=20, n_iter=5000)
    X_tsne = tsne.fit_transform(reps)

    def select_color(i):
        if i <= 8:
            return plt.cm.Set1(i)
        elif i <= 16:
            return plt.cm.Set1(i)
            # return plt.cm.Set2(i-9)
        elif i <= 28:
            return plt.cm.Set1(i)
            # return plt.cm.Set3(i-17)

    '''嵌入空间可视化'''
    y = [i//num_mode for i in range(X_tsne.shape[0])]
    if dim==2:
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(6, 6))

        ss = ['.' for _ in range(X_norm.shape[0]-20)]

        mscatter(X_norm[:-20, 0], X_norm[:-20, 1], c=range(X_norm.shape[0] - len(CLASSES_VOC)), s=100, cmap='rainbow',
                 m=ss)

        plt.scatter(X_norm[-20:, 0], X_norm[-20:, 1], color='r')

        for i in range(20):
            plt.text(X_norm[-20+i, 0], X_norm[-20+i, 1], CLASSES_VOC[i], color='r',
                     fontdict={'weight': 'bold', 'size': 9})
        # for i in range(X_norm.shape[0]):
        #     plt.text(X_norm[i, 0], X_norm[i, 1], CLASSES[y[i]], color=select_color(y[i]),
        #              fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
    elif dim==3:
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        # ss = []

        ss = ['.' for _ in range(X_norm.shape[0])]

        # for y in range(0, f_h):
        #     for x in range(0, f_w):
        #         ss.append(shapes[x])
        # plt.scatter(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], c=range(X_norm.shape[0]-num_cls), cmap='rainbow', marker=ss)
        # mscatter(X_norm[:, 0], X_norm[:, 1], zs=X_norm[:, 2], c=range(X_norm.shape[0] - num_cls), s=100, cmap='rainbow',
        #          m=ss)

        # for i in range(X_norm.shape[0]):
        #     # ax.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(y[i]), color='g',
        #     #          fontdict={'weight': 'bold', 'size': 9})
        #     ax.scatter(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], color='g')

        ax.scatter(X_norm[:-20, 0], X_norm[:-20, 1], X_norm[:-20, 2], color='g')
        ax.scatter(X_norm[-20:, 0], X_norm[-20:, 1], X_norm[-20:, 2], color='r')

        for i in range(20):
            ax.text(X_norm[-20+i, 0], X_norm[-20+i, 1], X_norm[-20+i, 2], CLASSES_VOC[i], color='r',
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

def show_dis_between_reps(reps, cls_ids=None):
    reps = reps.reshape(-1, reps.size(-1))
    class_names = CLASSES
    if cls_ids is not None:
        reps = reps[cls_ids, :]
        class_names = [class_names[i] for i in cls_ids]
    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = list(class_names)
    reps_exp1 = reps.unsqueeze(0).expand(num_cls, -1, -1)
    reps_exp2 = reps.unsqueeze(1).expand(-1, num_cls, -1)
    dis_mat = torch.sqrt(((reps_exp1-reps_exp2)**2).sum(-1))
    print('dis_mat.sum(): ', dis_mat.sum())
    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    plt.imshow(dis_mat, cmap='rainbow', norm=norm)
    plt.colorbar()
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    # fig.autofmt_xdate()
    plt.xticks(rotation=270)
    plt.show()

def show_rep_dims(reps, cls_ids=None):
    reps = reps.reshape(-1, reps.size(-1))
    class_names = CLASSES
    if cls_ids is not None:
        reps = reps[cls_ids, :]
        class_names = [class_names[i] for i in cls_ids]
    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = list(class_names)
    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.4)
    plt.imshow(torch.abs(reps), cmap='rainbow', norm=norm)
    plt.colorbar()
    plt.yticks(scale_ls, label_ls)
    plt.show()

def show_dim_dis_between_reps(reps, cls_id):
    reps = reps.reshape(-1, reps.size(-1))
    rep1 = reps[cls_id, :].expand_as(reps)
    dis_vector = ((rep1-reps)**2)

    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = list(CLASSES)

    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
    plt.imshow(dis_vector, cmap='rainbow', norm=norm)
    plt.colorbar()
    plt.yticks(scale_ls, label_ls)
    plt.show()

def show_reps(reps, novel_idx=1):
    novel_cls_ids = [CLASSES.index(cls_name) for cls_name in novel_sets[novel_idx-1]]
    base_cls_ids = list(set(range(len(CLASSES))) - set(novel_cls_ids))
    reps = reps.reshape(-1, reps.size(-1))
    novel_reps = reps[novel_cls_ids, :]
    base_reps = reps[base_cls_ids, :]

    reps = torch.cat([base_reps, novel_reps], dim=0)
    plt.imshow(reps, cmap='rainbow')
    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = [CLASSES[i] for i in (base_cls_ids+novel_cls_ids)]
    plt.yticks(scale_ls, label_ls)
    plt.show()

def show_emb_vectors():
    path = 'mytest/ga_dmlneg3_10s_t3s/ga_retina_dmlneg3_feature16.pth'
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    emb_vectors = checkpoint['emb_vectors'][4]
    reps = checkpoint['reps'][4]
    emb_vectors = emb_vectors.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1)
    print(emb_vectors.shape)
    emb_rep = torch.cat([emb_vectors, reps], dim=0)
    reps_visual(emb_rep, dim=3)

def show_dim_dis_sum(reps, cls_ids=None):
    reps = reps.reshape(-1, reps.size(-1))
    class_names = CLASSES
    if cls_ids is not None:
        reps = reps[cls_ids, :]
        class_names = [class_names[i] for i in cls_ids]

    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = list(class_names)

    reps_exp1 = reps.unsqueeze(0).expand(num_cls, -1, -1)
    reps_exp2 = reps.unsqueeze(1).expand(-1, num_cls, -1)
    dis_mat = ((reps_exp1-reps_exp2)**2).mean(dim=1)

    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    plt.imshow(dis_mat, cmap='rainbow', norm=norm)
    plt.colorbar()
    plt.yticks(scale_ls, label_ls)
    plt.show()

if __name__ == '__main__':
    # show_dis_between_reps(reps)
    # show_reps(reps, 2)
    # reps_visual(reps)
    # show_emb_vectors()
    # print((reps_ori / reps).var(-1))
    show_dis_between_reps(reps_ori, VOC_base_ids[0])
    show_rep_dims(reps, VOC_base_ids[0])
    # show_dim_dis_between_reps(reps, CLASSES.index('cat'))
    # show_dim_dis_sum(reps, VOC_base_ids[0])
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import torch

# checkpoint_file = 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_10_14_16_ind1_1/epoch_16.pth'
checkpoint_file = 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base2_r1_lr00025x2x2_10_14_16_ind1_1/epoch_16.pth'

checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
reps = checkpoint['state_dict']['bbox_head.representations']
# dim = 2
dim = 2

def reps_visual(reps, dim=2):
    '''t-SNE'''
    num_cls = reps.size(0)
    num_mode = reps.size(1)
    reps = reps.numpy().reshape(-1, reps.size(-1))
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=5, n_iter=300)
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
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], CLASSES[y[i]], color=select_color(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
    elif dim==3:
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        for i in range(X_norm.shape[0]):
            ax.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(y[i]), color=select_color(y[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
              ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
              ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]

def show_dis_between_reps(reps):
    reps = reps.reshape(-1, reps.size(-1))
    num_cls = reps.shape[0]
    scale_ls = range(num_cls)
    label_ls = list(CLASSES)
    reps_exp1 = reps.unsqueeze(0).expand(num_cls, -1, -1)
    reps_exp2 = reps.unsqueeze(1).expand(-1, num_cls, -1)
    dis_mat = torch.sqrt(((reps_exp1-reps_exp2)**2).sum(-1))
    fig = plt.figure()
    plt.imshow(dis_mat, cmap='rainbow')
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    fig.autofmt_xdate()
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

if __name__ == '__main__':
    # show_dis_between_reps(reps)
    show_reps(reps, 2)
    # reps_visual(reps)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import torch

# checkpoint_file = 'work_dirs/ga_dml_rpn_x101_uw20_embsize128_alpha015_lr0005-7-11-13_CE_ind1_beta004_tloss4_noeloss/epoch_13.pth'
# checkpoint_file = 'work_dirs/ga_dml_rpn_x101_voc_embsize128_alpha015_lr0005-7-13-17_CE_ind1_beta004/epoch_13.pth'
# checkpoint_file = 'work_dirs/ga_dml_rpn_x101_voc_embsize256_alpha015_lr000125-36-50-60_CE_ind1_beta004_tloss4_m3_t10s_aug/epoch_2.pth'
checkpoint_file = 'work_dirs/ga_dml_rpn2_x101_coco_n10s_r15_embsize1024_256_alpha015_lr000125-34-40-44_CE_ind1_beta004_tloss4_te_el025/epoch_42.pth'

checkpoint = torch.load(checkpoint_file)
reps = checkpoint['state_dict']['bbox_head.representations']
# dim = 2
dim = 2
'''t-SNE'''
num_cls = reps.size(0)
num_mode = reps.size(1)
reps = reps.numpy().reshape(-1, reps.size(-1))
tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
X_tsne = tsne.fit_transform(reps)

def select_color(i):
    if i <= 8:
        return plt.cm.Set1(i)
    elif i <= 16:
        return plt.cm.Set2(i-9)
    elif i <= 28:
        return plt.cm.Set3(i-17)

'''嵌入空间可视化'''
y = [i//num_mode for i in range(X_tsne.shape[0])]
if dim==2:
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(6, 6))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=select_color(y[i]),
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
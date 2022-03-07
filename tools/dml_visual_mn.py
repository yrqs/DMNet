import torch
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import numpy as np
import matplotlib.patches as mpathes
from matplotlib import rc
# rc('text', usetex=True)

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

file_outs_dict  = {
        'FPN' : ('fpn_outs.pth', ['laterals_ori', 'laterals', 'fpn_outs']),
        'DFPN' : ('dfpn_outs.pth', ['laterals_ori', 'laterals', 'cls_outs', 'reg_outs']),
        'DFPN2' : ('dfpn2_outs.pth', ['laterals_ori_cls', 'laterals_ori_reg', 'laterals_cls', 'laterals_reg', 'outs_cls', 'outs_reg']),
        'DFPN3' : ('dfpn3_outs.pth', ['laterals_ori', 'laterals', 'fpn_outs', 'spatial_attentions_cls', 'spatial_attentions_reg', 'outs_cls', 'outs_reg']),
        'DFPN4' : ('dfpn4_outs.pth', ['laterals', 'laterals', 'fpn_outs', 'attentions_cls', 'attentions_reg', 'outs_cls', 'outs_reg']),
        'ga_retina_dml' : ('ga_retina_dml_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'emb_vectors']),
        'ga_retina_dml2' : ('ga_retina_dml2_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance']),
        'ga_retina_dml2D' : ('ga_retina_dml2D_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance']),
        'ga_retina_dml3' : ('ga_retina_dml3_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance']),
        'ga_retina_dml3D' : ('ga_retina_dml3D_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance', 'emb_vectors', 'probs_bg']),
        'ga_retina_dml7' : ('ga_retina_dml7_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance', 'emb_vectors', 'probs_bg']),
        'ga_retina_dml14D' : ('ga_retina_dml14D_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance', 'probs_bg']),
        'ga_retina_dml16D' : ('ga_retina_dml16D_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_loc', 'cls_feat_enhance', 'probs_bg']),
        'ga_retina_dml24' : ('ga_retina_dml24_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp', 'cls_feat_enhance', 'reg_feat_enhance']),
        'ga_retina_dmlneg3' : ('ga_retina_dmlneg3_feature.pth', ['cls_scores', 'emb_vectors', 'reps', 'reps_neg', 'probs_ori']),
    }

config_file = 'configs/few_shot/voc/voc_test.py'

CLASSES_VOC = ('plane', 'bike', 'bird*', 'boat', 'bottle', 'bus*', 'car', 'cat',
               'chair', 'cow*', 'table', 'dog', 'horse', 'motorbike*', 'person',
               'pottedplant', 'sheep', 'sofa*', 'train', 'tv')

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

def show_emb_vectors_TSNE(outs, img=None, dim=2):

    def select_color(i):
        if i <= 8:
            return plt.cm.Set1(i)
        elif i <= 16:
            return plt.cm.Set2(i-9)
        elif i <= 28:
            return plt.cm.Set3(i-17)
    '''t-SNE'''
    emb_vectors_tuple = outs['emb_vectors']
    reps_tuple = outs['reps']
    cls_scores_tuple = outs['cls_scores']

    # plt.figure(1)
    plt.figure(1, figsize=(10, 6))
    # for i in range(len(reps_tuple)):
    for scale_idx in range(len(reps_tuple)-2, len(reps_tuple)):
        f_w = emb_vectors_tuple[scale_idx].size(3)
        f_h = emb_vectors_tuple[scale_idx].size(2)

        reps = reps_tuple[scale_idx]
        reps = reps.contiguous().view(-1, reps.size(-1))
        emb_vectors = emb_vectors_tuple[scale_idx]
        emb_vectors = emb_vectors[0]
        emb_vectors = emb_vectors.permute(1, 2, 0).contiguous()
        emb_vectors = emb_vectors.view(-1, emb_vectors.size(-1))

        vectors = torch.cat([emb_vectors, reps], 0)
        vectors = vectors.numpy()
        tsne = manifold.TSNE(n_components=2, init='pca', perplexity=20, n_iter=5000)
        X_tsne = tsne.fit_transform(vectors)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

        # X_norm = X_norm*0.5

        cls_scores = cls_scores_tuple[scale_idx]
        cls_scores = cls_scores[0]
        cls_scores = cls_scores.permute(1, 2, 0).contiguous()
        cls_scores = cls_scores.view(-1, cls_scores.size(-1))
        cls_scores = cls_scores.softmax(-1)
        cls_pre_score, cls_pre_idx = cls_scores.max(dim=-1, keepdim=True)
        num_cls = len(CLASSES_VOC)

        ss = []
        for y in range(0, f_h):
            for x in range(0, f_w):
                    ss.append(shapes[x])
        # plt.scatter(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], c=range(X_norm.shape[0]-num_cls), cmap='rainbow', marker=ss)
        mscatter(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], c=range(X_norm.shape[0]-num_cls), s=100, cmap='rainbow', m=ss)

        # plt.plot(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], 'r.')
        for i in range(X_norm.shape[0]-num_cls):
            cls_pre_name = '' if cls_pre_idx[i] == 0 else CLASSES_VOC[cls_pre_idx[i]-1]
            plt.text(X_norm[i, 0], X_norm[i, 1], cls_pre_name, color='black',
                     fontdict={'weight': 'bold', 'size': 13})
        plt.plot(X_norm[-num_cls:, 0], X_norm[-num_cls:, 1], 'g.')
        for i in range(num_cls):
            plt.text(X_norm[-(i+1), 0], X_norm[-(i+1), 1], CLASSES_VOC[-(i+1)], color=colors[i],
                     fontdict={'weight': 'bold', 'size': 13})
        plt.xticks([])
        plt.yticks([])

        if img is not None:
            print('plot img ...')
            plt.figure(2)
            plt.imshow(img)
            img_h = img.shape[0]
            img_w = img.shape[1]
            x_step = img_w / f_w
            y_step = img_h / f_h
            xs = []
            ys = []
            for y in range(0, f_h):
                for x in range(0, f_w):
                    xs.append((x+0.5)*x_step)
                    ys.append((y+0.5)*y_step)
            # plt.scatter(xs, ys, c=range(len(xs)), cmap='rainbow', marker=ss)
            mscatter(xs, ys, c=range(len(xs)), cmap='rainbow', m=ss, s=80)
            # viridis
            # for x in range(f_w):
            #     plt.plot([x*x_step, x*x_step], [0, img_h-1], 'r')
            # for y in range(f_h):
            #     plt.plot([0, img_w-1], [y*y_step, y*y_step], 'r')
            plt.xticks([])
            plt.yticks([])

def show_emb_vectors_TSNE_single(outs, scale_idx=-1, x_label=None, extra=None, is_show_text=True):
    emb_vectors_tuple = outs['emb_vectors']
    reps_tuple = outs['reps']
    reps_neg_tuple = outs['reps_neg']
    cls_scores_tuple = outs['cls_score']
    probs_ori_tuple = outs['probs_ori']
    # plt.figure(1)
    # plt.figure(1, figsize=(10, 6))
    # for i in range(len(reps_tuple)):
    f_w = emb_vectors_tuple[scale_idx].size(3)
    f_h = emb_vectors_tuple[scale_idx].size(2)

    reps = reps_tuple[scale_idx]
    reps = reps.contiguous().view(-1, reps.size(-1))
    reps_neg = reps_neg_tuple[scale_idx]
    num_modes_neg = reps_neg.shape[1]
    reps_neg = reps_neg.contiguous().view(-1, reps_neg.size(-1))
    emb_vectors = emb_vectors_tuple[scale_idx]
    emb_vectors = emb_vectors[0]
    emb_vectors = emb_vectors.permute(1, 2, 0).contiguous()
    emb_vectors = emb_vectors.view(-1, emb_vectors.size(-1))

    vectors = torch.cat([emb_vectors, reps, reps_neg], 0)
    # vectors = torch.cat([emb_vectors, reps], 0)
    vectors = vectors.numpy()
    tsne = manifold.TSNE(n_components=3, init='pca', perplexity=20, n_iter=1000)
    
    X_tsne = tsne.fit_transform(vectors)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    # X_norm = X_norm*0.5

    cls_scores = cls_scores_tuple[scale_idx]
    cls_scores = cls_scores[0]
    cls_scores = cls_scores.permute(1, 2, 0).contiguous()
    cls_scores = cls_scores.view(-1, cls_scores.size(-1))

    probs_ori = probs_ori_tuple[scale_idx]
    probs_ori = probs_ori[0]
    probs_ori = probs_ori.permute(1, 2, 0).contiguous()
    probs_ori = probs_ori.view(-1, probs_ori.size(-1))

    # cls_scores = cls_scores.softmax(-1)
    cls_pre_score, cls_pre_idx = cls_scores.max(dim=-1, keepdim=True)

    ss = []
    num_cls = len(CLASSES_VOC)
    for y in range(0, f_h):
        for x in range(0, f_w):
            ss.append(shapes[x])
    # plt.scatter(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], c=range(X_norm.shape[0]-num_cls), cmap='rainbow', marker=ss)
    mscatter(X_norm[:-num_cls*(num_modes_neg+1), 0], X_norm[:-num_cls*(num_modes_neg+1), 1], c=range(X_norm.shape[0] - num_cls*(num_modes_neg+1)), s=100, cmap='rainbow',
             m=ss)

    if is_show_text:
        for i in range(X_norm.shape[0] - num_cls * (num_modes_neg + 1)):
            cls_pre_name = '' if cls_pre_score[i] <= score_thresh else CLASSES_VOC[cls_pre_idx[i]]
            cls_pre_score_text = '' if cls_pre_score[i] < score_thresh else str(int(float(cls_pre_score[i]) * 100))
            cls_pre_prob_ori_text = '' if probs_ori[i, cls_pre_idx[i]] < score_thresh else '|' + str(int(float(probs_ori[i, cls_pre_idx[i]]) * 100))
            # cls_pre_score_text = ''
            # show_text = cls_pre_name + cls_pre_score_text + cls_pre_prob_ori_text
            show_text = cls_pre_name
            plt.text(X_norm[i, 0], X_norm[i, 1], show_text, color='k',
                     fontdict={'weight': 'bold', 'size': 20})
    plt.plot(X_norm[-num_cls * (num_modes_neg + 1):-num_cls * num_modes_neg, 0],
             X_norm[-num_cls * (num_modes_neg + 1):-num_cls * num_modes_neg, 1],
             marker='.', color='deeppink', linestyle='', markersize=10)
    plt.plot(X_norm[-num_cls * num_modes_neg:, 0], X_norm[-num_cls * num_modes_neg:, 1],
             marker='.', color='blue', linestyle='', markersize=10)
    if is_show_text:
        for i in range(num_cls * num_modes_neg, num_cls * (num_modes_neg + 1)):
            plt.text(X_norm[-(i + 1), 0], X_norm[-(i + 1), 1], CLASSES_VOC[-(i - num_cls * num_modes_neg + 1)],
                     color='deeppink',
                     fontdict={'weight': 'bold', 'size': 20})
        for i in range(0, num_cls * num_modes_neg):
            plt.text(X_norm[-(i + 1), 0], X_norm[-(i + 1), 1], CLASSES_VOC[-(i // num_modes_neg + 1)],
                     color='blue',
                     fontdict={'weight': 'bold', 'size': 20})
    plt.xticks([])
    plt.yticks([])

    plt.figure(3)
    mscatter(X_norm[:-num_cls*(num_modes_neg+1), 0], X_norm[:-num_cls*(num_modes_neg+1), 1], c=range(X_norm.shape[0] - num_cls*(num_modes_neg+1)), s=100, cmap='rainbow',
             m=ss)
    plt.plot(X_norm[-num_cls * (num_modes_neg + 1):-num_cls * num_modes_neg, 0],
             X_norm[-num_cls * (num_modes_neg + 1):-num_cls * num_modes_neg, 1],
             marker='.', color='deeppink', linestyle='', markersize=10)
    plt.plot(X_norm[-num_cls * num_modes_neg:, 0], X_norm[-num_cls * num_modes_neg:, 1],
             marker='.', color='blue', linestyle='', markersize=10)
    plt.xticks([])
    plt.yticks([])

    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    if x_label is not None:
        plt.xlabel(x_label, fontdict={'size': 20})

    if extra is not None:
        plt.subplot(1, 4, 4)
        mscatter(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], c=range(X_norm.shape[0] - num_cls), s=100, cmap='rainbow',
                 m=ss)

        # plt.plot(X_norm[:-num_cls, 0], X_norm[:-num_cls, 1], 'r.')
        for i in range(X_norm.shape[0] - num_cls):
            cls_pre_name = '' if cls_pre_idx[i] == 0 else CLASSES_VOC[cls_pre_idx[i] - 1]
            plt.text(X_norm[i, 0], X_norm[i, 1], cls_pre_name, color='k',
                 fontdict={'weight': 'bold', 'size': 20})
        plt.plot(X_norm[-num_cls:, 0], X_norm[-num_cls:, 1], 'g.')
        for i in range(num_cls):
            plt.text(X_norm[-(i + 1), 0], X_norm[-(i + 1), 1], CLASSES_VOC[-(i + 1)], color='deeppink',
                 fontdict={'weight': 'bold', 'size': 20})
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        if x_label is not None:
            plt.xlabel('Close-up inside the red rectangle', fontdict={'size': 20})

def show_emb_vectors_TSNE_multi_epoch(outs_list, scale_idx=-1, img=None, x_label_list=None, extra=None):
    emb_vectors_tuple = outs_list[0]['cls_feat']
    # emb_vectors_tuple = outs_list[0]['emb_vectors']
    # emb_vectors_tuple = outs_list[0]['cls_feat_enhance']
    f_w = emb_vectors_tuple[scale_idx].size(3)
    f_h = emb_vectors_tuple[scale_idx].size(2)

    if scale_idx < 0:
        scale_idx = len(emb_vectors_tuple) + scale_idx

    outs_num = len(outs_list)

    print('plot img ...')
    plt.figure(0)
    plt.subplot(1, outs_num+1, 1)
    plt.imshow(img)
    img_h = img.shape[0]
    img_w = img.shape[1]
    x_step = img_w / f_w
    y_step = img_h / f_h
    xs = []
    ys = []
    for y in range(0, f_h):
        for x in range(0, f_w):
            xs.append((x+0.5)*x_step)
            ys.append((y+0.5)*y_step)

    ss = []
    for y in range(0, f_h):
        for x in range(0, f_w):
            ss.append(shapes[x])

    mscatter(xs, ys, c=range(len(xs)), cmap='rainbow', m=ss, s=80)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Novel-class image (scale %d)' % (scale_idx+1), fontdict={'size': 20})

    for i, outs in enumerate(outs_list):
        ax = plt.subplot(1, outs_num+1, i+2)
        if i is len(outs_list) - 2:
            show_emb_vectors_TSNE_single(outs, scale_idx, x_label=x_label_list[i], extra=extra)
            if extra is not None:
                rect = mpathes.Rectangle((0.79, 0.79), 0.2, 0.2, color='r', fill=False, linewidth=1.5)
                ax.add_patch(rect)
                break
        else:
            show_emb_vectors_TSNE_single(outs, scale_idx, x_label=x_label_list[i], extra=None)

    plt.show()

def show_emb_vectors_TSNE_multi_scale(outs, scale_idx_list=None, img=None, x_label_list=None):
    if scale_idx_list is None:
        scale_idx_list = [-1]
    plt.figure(0)
    print('plot img ...')
    scale_num = len(scale_idx_list)
    for i, scale_idx in enumerate(scale_idx_list):
        emb_vectors_tuple = outs['emb_vectors']
        if scale_idx < 0 :
            scale_idx = len(emb_vectors_tuple) + scale_idx
        f_w = emb_vectors_tuple[scale_idx].size(3)
        f_h = emb_vectors_tuple[scale_idx].size(2)

        plt.subplot(1, scale_num*2, i*scale_num+1)
        plt.imshow(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        x_step = img_w / f_w
        y_step = img_h / f_h
        xs = []
        ys = []
        for y in range(0, f_h):
            for x in range(0, f_w):
                xs.append((x+0.5)*x_step)
                ys.append((y+0.5)*y_step)

        ss = []
        for y in range(0, f_h):
            for x in range(0, f_w):
                ss.append(shapes[x])

        mscatter(xs, ys, c=range(len(xs)), cmap='rainbow', m=ss, s=80)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Base-class image (scale %d)' % (scale_idx+1), fontdict={'size': 20})

        plt.subplot(1, scale_num*2, i*2+2)
        show_emb_vectors_TSNE_single(outs, scale_idx, x_label=x_label_list[i])
    plt.show()

def show_emb_vectors_TSNE_multi_img(outs_list, scale_idx_list=None, img_list=None, x_label_list=None):
    if scale_idx_list is None:
        scale_idx_list = [-1]
    plt.figure(0)
    print('plot img ...')
    img_num = len(img_list)
    for i, scale_idx in enumerate(scale_idx_list):
        img = img_list[i]
        outs = outs_list[i]
        emb_vectors_tuple = outs['emb_vectors']
        if scale_idx < 0 :
            scale_idx = len(emb_vectors_tuple) + scale_idx
        f_w = emb_vectors_tuple[scale_idx].size(3)
        f_h = emb_vectors_tuple[scale_idx].size(2)

        plt.subplot(1, img_num*2+1, i*img_num+1)
        plt.imshow(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        x_step = img_w / f_w
        y_step = img_h / f_h
        xs = []
        ys = []
        for y in range(0, f_h):
            for x in range(0, f_w):
                xs.append((x+0.5)*x_step)
                ys.append((y+0.5)*y_step)

        ss = []
        for y in range(0, f_h):
            for x in range(0, f_w):
                ss.append(shapes[x])

        mscatter(xs, ys, c=range(len(xs)), cmap='rainbow', m=ss, s=80)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Novel-class image (scale %d)' % (scale_idx+1), fontdict={'size': 20})

        plt.subplot(1, img_num*2+1, i*2+2)
        show_emb_vectors_TSNE_single(outs, scale_idx, x_label=x_label_list[i])

    plt.show()


def show_img_with_marks(img, outs, scale_idx):
    plt.imshow(img)
    emb_vectors_tuple = outs['emb_vectors']
    if scale_idx < 0:
        scale_idx = len(emb_vectors_tuple) + scale_idx
    f_w = emb_vectors_tuple[scale_idx].size(3)
    f_h = emb_vectors_tuple[scale_idx].size(2)
    img_h = img.shape[0]
    img_w = img.shape[1]
    x_step = img_w / f_w
    y_step = img_h / f_h
    xs = []
    ys = []
    for y in range(0, f_h):
        for x in range(0, f_w):
            xs.append((x + 0.5) * x_step)
            ys.append((y + 0.5) * y_step)

    ss = []
    for y in range(0, f_h):
        for x in range(0, f_w):
            ss.append(shapes[x])

    mscatter(xs, ys, c=range(len(xs)), cmap='rainbow', m=ss, s=80)
    plt.xticks([])
    plt.yticks([])

score_thresh = 0.3
scan_img = False
show_type = ['multi-img', 'multi-epoch', 'multi-scale', 'scan-image'][1]

show_list = [43, 44, 46, 47, ]
# show_list = [13, 16, 18, 32, 34, 38, 43, 44, 46, 47, ]
not_show_list = []

filter_show = True
filter_not_show = False

if __name__ == '__main__':
    cfg = Config.fromfile(config_file)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    feature_type = 'ga_retina_dmlneg3'
    root_path = 'mytest/ga_dmlneg3_10s_t3s/'

    file_name_base, outs_names = file_outs_dict[feature_type]
    file_name_base_path = root_path + file_name_base
    file_num = 50

    for i, data in enumerate(data_loader):
        if scan_img:
            continue
        if filter_show and i not in show_list:
            continue
        if filter_not_show and i in not_show_list:
            continue
        img = data['img'][0]
        img = img.squeeze(0)
        img = img.flip(dims=[2])
        if scan_img:
            plt.imshow(img)
            plt.xlabel('%d' % i)
            plt.show()
            continue

        file_name = file_name_base_path[:-4] + str(i + 1) + file_name_base_path[-4:]
        outs = torch.load(file_name, map_location=torch.device("cpu"))

        plt.rc('font', family='Times New Roman')

        x_label_list = ['Last epoch', 'Last epoch']
        scale_idx_list = [-2, -1]
        # show_emb_vectors_TSNE_multi_scale(outs, scale_idx_list=scale_idx_list, img=img, x_label_list=x_label_list)
        for scale_idx in scale_idx_list:
            plt.figure(1)
            show_img_with_marks(img, outs, scale_idx)
            plt.figure(2)
            show_emb_vectors_TSNE_single(outs, scale_idx, is_show_text=True)
            plt.show()
    exit()

    if show_type is 'multi-img':
        outs_list = []
        img_list = []

        feature_type = 'ga_retina_dml3'
        root_path = 'mytest/ga_retina_dml3_finetune_epoch_18_1shot/'

        file_name_base, outs_names = file_outs_dict[feature_type]
        file_name_base_path = root_path + file_name_base
        file_num = 17

        root_path_list = ['mytest/ga_retina_dml3_finetune_epoch_18_1shot/']
        file_name_base_path_list = [rp + file_name_base for rp in root_path_list]

        for i, data in enumerate(data_loader):
            if not scan_img and i not in [1, 16]:
                continue
            img = data['img'][0]
            img = img.squeeze(0)
            if scan_img:
                plt.imshow(img)
                plt.xlabel('%d' % i)
                plt.show()
                continue

            img_list.append(img)
            file_name = file_name_base_path[:-4] + str(i + 1) + file_name_base_path[-4:]
            outs = torch.load(file_name, map_location=torch.device("cpu"))
            outs_list.append(outs)
        x_label_list = ['Last epoch', 'Last epoch']
        scale_idx_list = [-1, -2]
        plt.rc('font', family='Times New Roman')
        show_emb_vectors_TSNE_multi_img(outs_list, scale_idx_list=scale_idx_list, img_list=img_list, x_label_list=x_label_list)
    elif show_type is 'multi-scale':
        feature_type = 'ga_retina_dml3'
        root_path = 'mytest/ga_retina_dml3_base_epoch_16_1shot/'

        file_name_base, outs_names = file_outs_dict[feature_type]
        file_name_base_path = root_path + file_name_base
        file_num = 17

        for i, data in enumerate(data_loader):
            if not scan_img and i not in [2]:
                continue
            img = data['img'][0]
            img = img.squeeze(0)
            if scan_img:
                plt.imshow(img)
                plt.xlabel('%d' % i)
                plt.show()
                continue

            file_name = file_name_base_path[:-4] + str(i + 1) + file_name_base_path[-4:]
            outs = torch.load(file_name, map_location=torch.device("cpu"))

            plt.rc('font', family='Times New Roman')

            x_label_list = ['Last epoch', 'Last epoch']
            scale_idx_list = [-2, -1]
            show_emb_vectors_TSNE_multi_scale(outs, scale_idx_list=scale_idx_list, img=img, x_label_list=x_label_list)
    elif show_type is 'multi-epoch':
        feature_type = 'ga_retina_dml3'

        file_name_base, outs_names = file_outs_dict[feature_type]
        file_num = 17

        root_path_list = ['mytest/ga_retina_dml3_base_epoch_16_1shot/',
                          'mytest/ga_retina_dml3_finetune_epoch_18_1shot/',
                          'mytest/ga_retina_dml3_finetune_epoch_18_1shot/']
        # root_path_list = ['mytest/ga_retina_dml3_base_epoch_0_1shot/',
        #                   'mytest/ga_retina_dml3_base_epoch_6_1shot/',
        #                   'mytest/ga_retina_dml3_base_epoch_16_1shot/']

        file_name_base_path_list = [rp + file_name_base for rp in root_path_list]

        outs_list = []
        img_list = []
        for i, data in enumerate(data_loader):
            if not scan_img and i not in [16]:
                continue
            img = data['img'][0]
            img = img.squeeze(0)
            if scan_img:
                plt.imshow(img)
                plt.xlabel('%d' % i)
                plt.show()
                continue

            outs_list = []
            for fnbp in file_name_base_path_list:
                file_name = fnbp[:-4] + str(i + 1) + fnbp[-4:]
                outs = torch.load(file_name, map_location=torch.device("cpu"))
                outs_list.append(outs)

            plt.rc('font', family='Times New Roman')

            # x_label_list = ['Initialization', 'Epoch 6', 'Epoch 16 (last epoch)']
            # x_label_list = ['Before fine-tuning', 'After fine-tuning']
            x_label_list = ['Before fine-tuning', 'After fine-tuning', 'Close-up inside the red rectangle']
            show_emb_vectors_TSNE_multi_epoch(outs_list, scale_idx=-2, img=img, x_label_list=x_label_list, extra=True)

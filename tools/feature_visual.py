import torch
import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import numpy as np

def show_feature_pyramid(feature_pyramid, name):
    print('=====================', name, '=====================')
    sub_row_num = 12
    sub_col_num = 22
    for feature_idx, feature in enumerate(feature_pyramid):
        print('level', feature_idx, ' start')
        for img_idx in range(feature.size(0)):
            print('image', img_idx, ' start')
            fig = plt.figure(name + str((feature_idx+1)*100+img_idx))
            for channel_idx in tqdm.tqdm(range(feature.size(1))):
                plt.subplot(sub_row_num, sub_col_num, channel_idx+1)
                im = plt.imshow(feature[img_idx, channel_idx].clone().cpu(), cmap='rainbow')
                plt.axis('off')
                # fig.colorbar(im)
    print('plotting...')
    plt.show()

def show_feature_pyramids(outs, names):
    for name in names:
        show_feature_pyramid(outs[name], name)

def show_feature_pyramids_L2(outs, names):
    subplot_row = len(names)
    subplot_col = 0
    for name in names:
        subplot_col = max(subplot_col, len(outs[name]))
    fig = plt.figure()
    for name_idx, name in enumerate(names):
        print(name, ' start')
        feature_pyramid = outs[name]
        for feature_idx, feature in enumerate(feature_pyramid):
            # print('level', feature_idx, ' start')
            if name is 'cls_loc':
                feature_L2 = feature.sigmoid().squeeze(1)
            elif name in ['probs_bg', 'spatial_attentions_cls', 'spatial_attentions_reg']:
                feature_L2 = feature.squeeze(1)
            elif name is 'emb_vectors':
                feature_L2 = torch.norm(feature, p=2, dim=1)
            else:
                feature_L2 = torch.norm(feature.sigmoid(), p=2, dim=1)
            plt.subplot(subplot_row, subplot_col, feature_idx+1 + name_idx*subplot_col)
            if feature_idx == 0:
                plt.ylabel(name)
            im = plt.imshow(feature_L2[0].clone().cpu(), cmap='rainbow')
            # plt.axis('off')
            fig.colorbar(im)
    print('plotting...')
    plt.show()

def show_feature_pyramid_L2(feature_pyramid, name):
    print('=====================', name, '=====================')
    feature_count = len(feature_pyramid)
    for img_idx in range(feature_pyramid[0].size(0)):
        print('image', img_idx, ' start')
        plt.figure(str(img_idx) + ':' + name)
        for feature_idx, feature in enumerate(feature_pyramid):
            print('level', feature_idx, ' start')
            feature_L2 = torch.norm(feature, p=2, dim=1)
            plt.subplot(1, feature_count, feature_idx+1)
            plt.imshow(feature_L2[img_idx].clone().cpu(), cmap='rainbow')
            plt.axis('off')
    print('plotting...')
    plt.show()

def reps_visual(reps, dim=2):
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

def show_emb_vectors_TSNE(outs, dim=2):

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

    plt.figure(figsize=(6, 6))
    # for i in range(len(reps_tuple)):
    for i in range(len(reps_tuple)-1, len(reps_tuple)):
        reps = reps_tuple[i]
        reps = reps.contiguous().view(-1, reps.size(-1))
        emb_vectors = emb_vectors_tuple[i]
        emb_vectors = emb_vectors[0]
        emb_vectors = emb_vectors.permute(1, 2, 0).contiguous()
        emb_vectors = emb_vectors.view(-1, emb_vectors.size(-1))

        vectors = torch.cat([emb_vectors, reps], 0)
        print(vectors.size())
        vectors = vectors.numpy()
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=101)
        X_tsne = tsne.fit_transform(vectors)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_tsne = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.plot(X_tsne[:-20, 0], X_tsne[:-20, 1], 'r.')
        plt.plot(X_tsne[-20:, 0], X_tsne[-20:, 1], 'g.')

        # reps = reps.numpy().reshape(-1, reps.size(-1))
        # tsne_rep = manifold.TSNE(n_components=3, init='pca', random_state=101)
        # X_tsne_rep = tsne_rep.fit_transform(reps)
        # x_min_rep, x_max_rep = X_tsne_rep.min(0), X_tsne_rep.max(0)
        #
        # emb_vectors = emb_vectors.numpy().reshape(-1, emb_vectors.size(-1))
        # tsne_emb = manifold.TSNE(n_components=3, init='pca', random_state=101)
        # X_tsne_emb = tsne_emb.fit_transform(emb_vectors)
        # x_min_emb, x_max_emb = X_tsne_emb.min(0), X_tsne_emb.max(0)
        #
        # X_norm_emb = (X_tsne_emb - x_min) / (x_max - x_min)  # 归一化
        # X_norm_rep = (X_tsne_rep - x_min) / (x_max - x_min)  # 归一化
        # plt.plot(X_norm_rep[:, 0], X_norm_rep[:, 1], 'r.')
        # plt.plot(X_norm_emb[:, 0], X_norm_emb[:, 1], 'g.')

        plt.xticks([])
        plt.yticks([])
        plt.show()


def distance(a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=-1))

def show_emb_vectors(outs):
    emb_vectors = outs['emb_vectors']
    for emb_vector_idx, emb_vector in enumerate(emb_vectors):
        emb_vector = emb_vector[0].permute(1, 2, 0).contiguous().cpu()
        dis = distance(emb_vector[0, 0], emb_vector[emb_vector.size(0)//2, emb_vector.size(1)//2])
        print(dis)

def show_cls_feat_enhances(outs):
    cls_feat_enhances = outs['cls_feat_enhance']
    fig = plt.figure()
    sub_row_num = len(cls_feat_enhances)
    sub_col_num = cls_feat_enhances[0].size(1)
    for cls_feat_enhance_idx, cls_feat_enhance in enumerate(cls_feat_enhances):
        # cls_feat_enhance = cls_feat_enhance.squeeze(0).sigmoid()
        cls_feat_enhance = cls_feat_enhance.squeeze(0).relu()
        for i in range(cls_feat_enhance.size(0)):
            cfe = cls_feat_enhance[i]
            plt.subplot(sub_row_num, sub_col_num, sub_col_num*cls_feat_enhance_idx + i+1)
            im = plt.imshow(cfe.clone().cpu(), cmap='rainbow')
            # plt.axis('off')
            fig.colorbar(im)
    plt.show()

def show_cls_feat_enhances_sum(outs):
    cls_feat_enhances = outs['cls_feat_enhance']
    fig = plt.figure()
    sub_row_num = len(cls_feat_enhances)
    sub_col_num = cls_feat_enhances[0].size(1)
    for cls_feat_enhance_idx, cls_feat_enhance in enumerate(cls_feat_enhances):
        # cls_feat_enhance = cls_feat_enhance.squeeze(0).sigmoid()
        cls_feat_enhance = cls_feat_enhance.squeeze(0).relu()
        cls_feat_enhance_l2 = torch.sqrt((cls_feat_enhance**2).sum(0))
        im = plt.imshow(cls_feat_enhance_l2.clone().cpu(), cmap='rainbow')
        plt.show()

def show_offsets(outs):
    offsets_cls = outs['offsets_cls']
    # offsets_cls = outs['offsets_cls']
    offsets_reg = outs['offsets_reg']
    offsets_cls_1 = []
    for o in offsets_cls:
        offsets_cls_1.append(torch.split(o, 9, dim=1)[0])

    for o in offsets_cls_1:
        print(o.size())

    fig = plt.figure()
    sub_row_num = len(offsets_cls_1)
    sub_col_num = offsets_cls_1[0].size(1)
    for offset_cls_idx, offsets_cls in enumerate(offsets_cls_1):
        offsets_cls = offsets_cls.squeeze(0)
        for i in range(offsets_cls.size(0)):
            cfe = offsets_cls[i]
            plt.subplot(sub_row_num, sub_col_num, sub_col_num*offset_cls_idx + i+1)
            im = plt.imshow(cfe.clone().cpu(), cmap='rainbow')
            # plt.axis('off')
            fig.colorbar(im)
    plt.show()

def show_reg_feat_enhances(outs):
    reg_feat_enhances = outs['reg_feat_enhance']
    fig = plt.figure()
    sub_row_num = len(reg_feat_enhances)
    sub_col_num = reg_feat_enhances[0].size(1)
    for cls_feat_enhance_idx, cls_feat_enhance in enumerate(reg_feat_enhances):
        cls_feat_enhance = cls_feat_enhance.squeeze(0).sigmoid()
        for i in range(cls_feat_enhance.size(0)):
            cfe = cls_feat_enhance[i]
            plt.subplot(sub_row_num, sub_col_num, sub_col_num*cls_feat_enhance_idx + i+1)
            im = plt.imshow(cfe.clone().cpu(), cmap='rainbow')
            # plt.axis('off')
            fig.colorbar(im)
    plt.show()


def similar_feature(feature1, feature2):
    # if feature1.shape[0] != 1:
    #     print('warning')
    _feature1 = feature1
    _feature2 = feature2
    _feature1 = _feature1.view(_feature1.shape[0]*_feature1.shape[1], -1)  # 将特征转换为(N*C)*(W*H)，即两维
    _feature2 = _feature2.view(_feature2.shape[0]*_feature2.shape[1], -1)

    _feature1 = _feature1.cpu().numpy()
    _feature2 = _feature2.cpu().numpy()
    similarity = _feature1.dot(_feature2.T).diagonal() / (1e-20 + ((np.sqrt(np.sum(_feature1 * _feature1, axis=1))) * np.sqrt(np.sum(_feature2 * _feature2, axis=1))))
    return similarity

def similar_feature_per_channel(x, y, norm=False):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos = num / (denom + 1e-30)
    if norm:
        sim = 0.5 + 0.5 * cos
    else:
        sim = cos
    return sim
# file = './similars/results_ga_retina_voc_base1.txt'

def feature_pyramid_similarity(feature_pyramid1, feature_pyramid2, norm=False):
    similarity = []
    for c in range(feature_pyramid1.size(1)):
        similarity.append(similar_feature_per_channel(feature_pyramid1[:, c, :, :], feature_pyramid2[:, c, :, :], norm))
    similarity = np.hstack(tuple(similarity))
    return similarity

def feature_pyramids_similarity(feature_pyramids1, feature_pyramids2, norm=False):
    similarity = []
    for fp1, fp2 in zip(feature_pyramids1, feature_pyramids2):
        similarity.append(feature_pyramid_similarity(fp1, fp2, norm=norm))
    return similarity

def plot_similarity(similarity):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots(1)
    im = ax.imshow(similarity, cmap='rainbow', norm=norm)  # 用cmap设置配色方案
    # im = ax.imshow(similarity, cmap=plt.cm.hot, norm=norm)  # 用cmap设置配色方案
    fig.colorbar(im)
    plt.show()

def plot_feature_pyramid_similarity(feature_pyramid1, feature_pyramid2):
    similarity = []
    for c in range(feature_pyramid1.size(1)):
        # print(feature_pyramid1.size())
        # similarity.append(similar_feature(feature_pyramid1[:, c, :, :], feature_pyramid2[:, c, :, :]))
        similarity.append(similar_feature_per_channel(feature_pyramid1[:, c, :, :], feature_pyramid2[:, c, :, :]))
            # print(similarity[-1])
    similarity = np.hstack(tuple(similarity))
    plot_similarity(similarity)

def plot_feature_pyramids_similarity(feature_pyramids1, feature_pyramids2):
    similarity = []
    for fp1, fp2 in zip(feature_pyramids1, feature_pyramids2):
        similarity.append(feature_pyramid_similarity(fp1, fp2))
    similarity = np.vstack(similarity)
    plot_similarity(similarity)

CLASSES_VOC = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


show_list = [13, 43, 44, 46, 47, ]
# show_list = [13, 16, 18, 32, 34, 38, 43, 44, 46, 47, ]
not_show_list = []

filter_show = True
filter_not_show = False

if __name__ == '__main__':
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
        'ga_retina_dmlneg3' : ('ga_retina_dmlneg3_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp']),
        'ga_retina_dmlneg7' : ('ga_retina_dmlneg7_feature.pth', ['cls_feat', 'reg_feat', 'cls_feat_adp', 'reg_feat_adp']),
    }
    # file_name = 'mytest/dfpn_outs1.pth'
    # outs_names = ['laterals', 'cls_outs', 'reg_outs']

    # file_name = 'mytest/fpn_outs_crop.pth'
    # outs_names = ['laterals', 'fpn_outs']

    # file_name = 'mytest/dfpn_outs_bird.pth'
    # outs_names = ['laterals', 'cls_outs', 'reg_outs']

    # outs = torch.load(file_name)

    # show_feature_pyramids(outs, outs_names)
    # file_name_base = 'mytest/dfpn_outs.pth'
    # file_name_base = 'mytest/dfpnv2_outs.pth'
    # outs_names = ['laterals', 'cls_outs', 'reg_outs']
    # file_name_base = 'mytest/retinaD_feature.pth'
    # file_name_base = 'mytest/retina_feature.pth'
    # outs_names = ['cls_feat', 'reg_feat']
    # file_name_base = 'mytest/gas_retinaD_feature.pth'
    # outs_names = ['cls_feat', 'reg_feat', 'loc_pred']
    # file_name_base = 'mytest/dfpnv3_outs.pth'
    # outs_names = ['laterals', 'cls_outs', 'reg_outs', 'att_cls', 'att_reg', 'cls_outs_att', 'reg_outs_att']
    # file_name = 'mytest/fpn_outs_crop.pth'
    # outs_names = ['laterals', 'fpn_outs']

    # feature_type = 'ga_retina_dml3'
    # feature_type = 'ga_retina_dmlneg7'
    feature_type = 'ga_retina_dmlneg3'
    # feature_type = 'DFPN4'
    # root_path = 'mytest/ga_dmlneg7_base2_t3s/'
    root_path = 'mytest/ga_dmlneg3_base2_t3s/'
    # root_path = 'mytest/ga_retina_dml3_dfpn2_1shot/'
    # root_path = 'mytest/ga_retina_dml2D_fpn_1shot/'
    # root_path = 'mytest/ga_retina_dml3_fpn_1shot/'

    file_name_base, outs_names = file_outs_dict[feature_type]
    file_name_base = root_path + file_name_base
    file_num = 50
    feature_pyramids_similarity_list = []
    for i in range(file_num):
        if filter_show and i not in show_list:
            continue
        file_name = file_name_base[:-4] + str(i+1) + file_name_base[-4:]
        outs = torch.load(file_name, map_location=torch.device("cpu"))
        # show_feature_pyramids_L2(outs, outs_names)
        # plot_feature_pyramids_similarity(outs['outs_cls'], outs['outs_reg'])
        # show_cls_feat_enhances(outs)
        show_cls_feat_enhances_sum(outs)
        # show_emb_vectors_TSNE(outs)
        # show_offsets(outs)
        # show_reg_feat_enhances(outs)
        # show_feature_pyramids(outs, outs_names)
        # show_emb_vectors(outs)
        # feature_pyramids_similarity_list.append(feature_pyramids_similarity(outs['outs_cls'], outs['outs_reg'], norm=False))
    #     feature_pyramids_similarity_list.append(feature_pyramids_similarity(outs['cls_feat'], outs['reg_feat']))
    # level_num = len(feature_pyramids_similarity_list[0])
    # for i in range(level_num):
    #     fps = [feature_pyramids_similarity_list[idx][i] for idx in range(len(feature_pyramids_similarity_list))]
    #     sim_level = np.vstack(fps)
    #     plot_similarity(sim_level)

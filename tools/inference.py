from mmdet.apis import init_detector, show_result
from mmdet.apis.inference import LoadImage

from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import os
import torch.nn.functional as F

import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt
import math

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

def frcn_roi_cls(detector, img, return_spicial_att=False):
    with torch.no_grad():
        cfg = detector.cfg
        device = next(detector.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        input_x = data['img'][0]
        x_c3 = detector.backbone(input_x)
        rois = torch.tensor([[0., 0, 0, input_x.size(3) - 1, input_x.size(2) - 1]]).to(input_x.device)
        x_pool = detector.bbox_roi_extractor(x_c3, rois.float())
        x_c5 = detector.shared_head(x_pool)
        bs, c, w, h = x_c5.shape
        spicial_att = detector.bbox_head.spicial_att(x_c5).reshape(bs, 1, -1).softmax(-1)
        spicial_att_res = spicial_att.reshape(bs, 1, 7, 7)
        x_c5 = x_c5.reshape(bs, c, -1)
        x_c5 = x_c5 * spicial_att
        x_vector = x_c5.sum(-1)
        if return_spicial_att:
            return x_vector, spicial_att_res
        else:
            return x_vector

def frcn_rois_cls(detector, img, roi_offsets):
    with torch.no_grad():
        cfg = detector.cfg
        device = next(detector.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        input_x = data['img'][0]
        x_c3 = detector.backbone(input_x)
        w = input_x.size(3) - 1
        h = input_x.size(2) - 1
        rois = []
        rois.append(torch.tensor([[0., 0, 0, w, h]]).to(input_x.device))
        for offset in roi_offsets:
            delta_x1, delta_x2, delta_y1, delta_y2 = offset
            rois.append(torch.tensor([[0., w * delta_x1, h * delta_y1, w + w * delta_x2, h + h * delta_y2]]).to(input_x.device))
        rois = torch.cat(rois, dim=0)
        x_pool = detector.bbox_roi_extractor(x_c3, rois.float())
        x_c5 = detector.shared_head(x_pool)
        bs, c, w, h = x_c5.shape
        spicial_att = detector.bbox_head.spicial_att(x_c5).reshape(bs, 1, -1).softmax(-1)
        x_c5 = x_c5.reshape(bs, c, -1)
        x_c5 = x_c5 * spicial_att
        x_vectors = x_c5.sum(-1)
        return x_vectors

def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def multi_scale(detector):
    # instance_root = 'mytest/voc_instances/test_3s/cow'
    instance_root = 'mytest/voc_instances/trainval_10shot/bottle'
    img_scales = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    res = []
    img = os.path.join(instance_root, '5.jpg')
    for img_scale in img_scales:
        detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
        res.append(frcn_roi_cls(detector, img))
    return res

def multi_img(detector):
    # instance_root = 'mytest/voc_instances/test_3s/cow'
    instance_root = 'mytest/voc_instances/trainval_10shot/bottle'
    img_scale = (256, 256)
    res = []
    detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
    att = []
    img_names = ['{}.jpg'.format(i) for i in range(10)]
    for img_name in img_names:
        img = os.path.join(instance_root, img_name)
        vector, _att = frcn_roi_cls(detector, img, return_spicial_att=True)
        res.append(vector)
        att.append(_att)
    return res, att

def multi_class(detector):
    instance_root = 'mytest/voc_instances/trainval_10shot'
    # class_names = ['cow', 'sheep', 'horse']
    class_names = CLASSES
    # img_scale = (128, 128)
    img_scale = (256, 256)
    res = []
    detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
    img_name = '0.jpg'
    for class_name in class_names:
        img = os.path.join(instance_root, class_name, img_name)
        res.append(frcn_roi_cls(detector, img))
    return res

def multi_class_avg(detector):
    instance_root = 'mytest/voc_instances/test_3s'
    # class_names = ['cow', 'sheep', 'horse']
    class_names = CLASSES
    img_scale = (128, 128)
    res = []
    detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
    img_name = '3.jpg'
    for class_name in class_names:
        img = os.path.join(instance_root, class_name, img_name)
        res.append(frcn_roi_cls(detector, img))
    return res

def show_vectors(vectors):
    feat_list = torch.split(vectors, 128, 1)
    for i, feat in enumerate(feat_list):
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        plt.imshow(feat, cmap='rainbow', norm=norm)
        plt.colorbar()
        plt.title('%d - %d' % (i*128, (i+1)*128-1))
        plt.show()

def multi_rois(detector):
    instance_root = 'mytest/voc_instances/trainval_10shot/cow'
    img_scale = (256, 256)
    detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
    img = os.path.join(instance_root, '0.jpg')
    roi_offsets = [
        (0, 0, -0.1, -0.1),
        (0, 0, -0.2, -0.2),
        (0, 0, -0.3, -0.3),
        (0, 0, -0.4, -0.4),
        (0, 0, -0.5, -0.5),
        (0, 0, -0.6, -0.6),
        (0, 0, -0.7, -0.7),
    ]
    return frcn_rois_cls(detector, img, roi_offsets)

def show_channels(vectors, norm_min=0., norm_max=1., class_names=None):
    feat_list = torch.split(vectors, 128, 1)
    for i, feat in enumerate(feat_list):
        norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max)
        plt.imshow(feat.clone().cpu(), cmap='rainbow', norm=norm)
        plt.colorbar()
        if class_names is not None:
            num_cls = len(class_names)
            scale_ls = range(num_cls)
            label_ls = list(class_names)
            plt.yticks(scale_ls, label_ls)
        plt.title('%d - %d' % (i*128, (i+1)*128-1))
        plt.show()

def plot_att(att):
    n = att.size(0)
    att = att.squeeze(1)
    plt.figure()
    row = int(n**0.5)
    col = math.ceil(n / row)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    for i in range(n):
        plt.subplot(row, col, i + 1)
        plt.imshow(att[i].cpu(), cmap='rainbow', norm=norm)
        plt.xlabel('{}'.format(i))



if __name__ == '__main__':
    # config = 'configs/few_shot/voc/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/finetuneDF.py'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/aug2/DF/10shot/epoch_16.pth'
    # config = 'configs/few_shot/voc/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/amsoftmax/finetuneDF.py'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/amsoftmax/DF/10shot/epoch_16.pth'
    # config = 'configs/few_shot/voc/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/finetuneDF.py'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/DF/10shot/epoch_16.pth'
    # config = 'configs/few_shot/voc/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/finetuneDF.py'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/DF/10shot/epoch_16.pth'
    # config = 'configs/few_shot/voc/frcn_r101_voc/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/margin_01/finetuneDFCrop.py'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/margin_01/DFCrop/10shot/epoch_16.pth'
    config = 'configs/few_shot/voc/frcn_r101_voc/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/split2/finetuneD.py'
    checkpoint = 'work_dirs/frcn_r101_voc/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/split2/D/10shot/epoch_16.pth'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/triplet_loss/margin_01/base/epoch_12.pth'
    # checkpoint = 'work_dirs/frcn_r101_voc_split1/torchvision/fs_bbox_head/wo_detach/1000_600/use_cos/DF/10shot/epoch_16.pth'
    detector = init_detector(config, checkpoint)

    # res = multi_scale(detector)
    # res, att = multi_img(detector)
    # res = multi_class(detector)
    # res = torch.cat(res, dim=0).detach()
    # att = torch.cat(att, dim=0).detach()
    # plot_att(att)

    with torch.no_grad():
        res = multi_rois(detector)

    fc_reg_w = detector.bbox_head.fc_reg.weight.detach()
    cls_att = torch.sub(1., fc_reg_w / fc_reg_w.max(dim=1, keepdim=True)[0])
    cls_att = cls_att.min(dim=0, keepdim=True)[0]
    res_att_norm = F.normalize((res * cls_att).detach(), dim=1, p=2)
    res_norm = F.normalize((res).detach(), dim=1, p=2)
    cls_id = CLASSES.index('bottle') + 1
    # fc_cls_w = F.normalize(detector.bbox_head.fc_cls.weight.detach()[cls_id:cls_id+1, :], dim=1, p=2)
    fc_cls_w = detector.bbox_head.fc_cls.weight.detach()[cls_id:cls_id+1, :]
    # fc_cls_w = detector.bbox_head.fc_cls.weight.detach()[1:, :]
    # plt.figure()
    # show_channels(
    #     torch.cat([
    #         # res,
    #         res_norm / res_norm.max(dim=1, keepdim=True)[0],
    #         # res_att_norm / res_att_norm.max(dim=1, keepdim=True)[0],
    #         # res_norm.mean(dim=0, keepdim=True).detach(),
    #         # res_norm.var(dim=0, keepdim=True).detach(),
    #         # res.mean(dim=0, keepdim=True).detach(),
    #         # res.var(dim=0, keepdim=True).detach(),
    #         # F.normalize(detector.bbox_head.fc_cls.weight.detach()[1:, :], dim=1, p=2),
    #         # fc_cls_w / fc_cls_w.max(dim=1, keepdim=True)[0],
    #         fc_cls_w / fc_cls_w.max(dim=1, keepdim=True)[0],
    #         # fc_cls_w * 1,
    #         fc_reg_w / fc_reg_w.max(dim=1, keepdim=True)[0]
    # ], dim=0),
    #     norm_min=0,
    #     norm_max=1.,
    #     # class_names=[str(i) for i in range(res.size(0))] + ['mean', 'var'] + list(CLASSES) + ['x', 'y', 'w', 'h'])
    #     # class_names=[str(i) for i in range(res.size(0))] + [str(i) for i in range(res.size(0))] + ['mean', 'var'] + ['cow'] + ['x', 'y', 'w', 'h'])
    #     # class_names=[str(i) for i in range(res.size(0))] + ['mean', 'var'] + ['cow'] + ['x', 'y', 'w', 'h'])
    #     class_names=[str(i) for i in range(res.size(0))] + ['cow'] + ['x', 'y', 'w', 'h'])
    #     # class_names=['cow'] + ['mean', 'var'] + list(CLASSES) + ['x', 'y', 'w', 'h'])

    with torch.no_grad():
        fg_w_norm = F.normalize(detector.bbox_head.fc_cls.weight[1:, :], p=2, dim=1)
        cls_feat_norm = F.normalize(res, p=2, dim=1)
        fg_w_norm_ex = fg_w_norm[None, :, :].expand(res.size(0), -1, -1)
        cls_feat_norm_ex = cls_feat_norm[:, None, :].expand_as(fg_w_norm_ex)
        fg_cls_score = (fg_w_norm_ex * cls_feat_norm_ex).sum(-1) * detector.bbox_head.cos_scale
        bg_cls_score = (detector.bbox_head.fc_cls.weight[0, :][None, :].expand_as(res) * res).sum(-1, keepdim=True)
        cls_score = torch.cat([bg_cls_score, fg_cls_score], dim=1).softmax(-1)[:, 1:]
        # cls_score = (fg_w_norm_ex * cls_feat_norm_ex).sum(-1)

    # cls_score = detector.bbox_head.fc_cls(res).softmax(-1)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    plt.figure()
    plt.imshow(cls_score.cpu(), cmap='rainbow', norm=norm)
    # class_names = ('bg', ) + CLASSES
    class_names = CLASSES
    num_cls = len(class_names)
    scale_ls = range(num_cls)
    label_ls = list(class_names)
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    # plt.yticks(range(num_cls-1), list(class_names[1:]))
    plt.xticks(rotation=270)
    plt.colorbar()
    # plt.show()

    res_ex1 = res[None, :, :].expand(res.size(0), -1, -1)
    res_ex2 = res[:, None, :].expand_as(res_ex1)
    cos_sim_mat = F.cosine_similarity(res_ex1, res_ex2, dim=2)

    plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    plt.imshow(cos_sim_mat.cpu(), cmap='rainbow', norm=norm)
    num_cls = len(CLASSES)
    scale_ls = range(num_cls)
    label_ls = list(CLASSES)
    plt.xticks(scale_ls, label_ls)
    plt.yticks(scale_ls, label_ls)
    plt.xticks(rotation=270)
    plt.colorbar()
    plt.show()

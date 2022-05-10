from mmdet.apis import init_detector, show_result
from mmdet.apis.inference import LoadImage

from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import os
import torch.nn.functional as F

import argparse

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

voc_novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
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

split_name_to_cls_ids = {
    'voc': {
        'base1' : VOC_base_ids[0],
        'base2' : VOC_base_ids[1],
        'base3' : VOC_base_ids[2],
        'novel1': VOC_novel_ids[0],
        'novel2': VOC_novel_ids[1],
        'novel3': VOC_novel_ids[2],
    }
}

DATASETS = {
    'voc': VOC_CLASSES
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='pre-trained checkpoint')
    parser.add_argument('dataset_name', help='dataset name')
    parser.add_argument('split_name', help='split name')
    parser.add_argument('--shot', type=int, default=None, help='n_shot')
    parser.add_argument('--seedn', type=int, default=None, help='nth seed')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    config = args.config
    checkpoint = args.checkpoint
    dataset_name = args.dataset_name
    split_name = args.split_name
    detector = init_detector(config, checkpoint)
    if split_name not in split_name_to_cls_ids[dataset_name].keys():
        raise ValueError('split_name is illegal')
    cls_ids = split_name_to_cls_ids[dataset_name][split_name]
    all_classes = DATASETS[dataset_name]
    init_class_names = [all_classes[id] for id in cls_ids]
    all_res = []
    with torch.no_grad():
        for class_name in init_class_names:
            instance_root = 'mytest/voc_instances/trainval_10shot/' + class_name
            for root, dirs, files in os.walk(instance_root):
                res_per_cls = []
                for file in files:
                    img = os.path.join(instance_root, file)
                    img_scales = [(128, 128), (256, 256)]
                    res = []
                    for img_scale in img_scales:
                        detector.cfg.data.test.pipeline[1]['img_scale'] = img_scale
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
                        res.append(detector.finetune_init(input_x))
                    res_per_cls.append(torch.cat(res, dim=0)[None, :, :])
            all_res.append(torch.cat(res_per_cls, dim=0)[None, :, :, :])
        all_res = torch.cat(all_res, dim=0)
        all_res = F.normalize(all_res, p=2, dim=-1)
        channels = all_res.shape[-1]
        all_res = all_res.reshape(len(cls_ids), -1, channels)
        all_res = all_res.mean(dim=1)

    model = torch.load(checkpoint, map_location=torch.device("cpu"))
    fc_cls_w = model['state_dict']['bbox_head.fc_cls.weight']
    for i in range(len(cls_ids)):
        fc_cls_w[cls_ids[i]+1, :] = all_res[i, :].cpu().data
    model['state_dict']['bbox_head.fc_cls.weight'] = fc_cls_w
    new_checkpoint = checkpoint[:-4] + '_finetune_init' + checkpoint[-4:]
    torch.save(model, new_checkpoint)

if __name__ == '__main__':
    main()
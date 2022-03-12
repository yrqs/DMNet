import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.datasets.builder import build_dataset

import tqdm
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['dataset']['pipeline'] = [
        x for x in train_data_cfg.dataset.pipeline if x['type'] not in skip_type
    ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    print(cfg)
    dataset = build_dataset(cfg.data.train)

    # progress_bar = mmcv.ProgressBar(len(dataset))
    valid_inds = []
    have_label_list = [0] * 21
    for item in tqdm.tqdm(dataset):
        # filename = os.path.join(args.output_dir,
        #                         Path(item['filename']).name
        #                         ) if args.output_dir is not None else None
        # mmcv.imshow_det_bboxes(
        #     item['img'],
        #     item['gt_bboxes'],
        #     item['gt_labels'] - 1,
        #     class_names=dataset.CLASSES,
        #     show=not args.not_show,
        #     out_file=filename,
        #     wait_time=args.show_interval)
        # progress_bar.update()
        labels = item['gt_labels']
        filename = item['img_info']['filename']
        ind = filename.split('/')[-1].split('.')[0]

        have_label_set = set()
        temp_list = [0] * 21
        for l in labels:
            if have_label_list[l] + temp_list[l] >= 3:
                have_label_set = set()
                break
            temp_list[l] += 1
            have_label_set.add(l)
        if len(have_label_set) > 0:
            for idx, label_num in enumerate(temp_list):
                have_label_list[idx] += label_num
            valid_inds.append(ind)
        ok = True
        for hl_i, hl in enumerate(have_label_list):
            if hl_i == 0:
                continue
            if hl < 3:
                ok = False
                break
        if ok:
            print(have_label_list)
            print(valid_inds)
            with open('mytest/test_3shot.txt', 'w') as f:
                for vi in valid_inds:
                    f.write(vi+'\n')
            return
    print(valid_inds)

def main1():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    print(cfg)
    dataset = build_dataset(cfg.data.train)

    # progress_bar = mmcv.ProgressBar(len(dataset))
    for item in tqdm.tqdm(dataset):
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        # cv2.imshow('1', item['img']/255)
        # cv2.waitKey(2000)
        mmcv.imshow_det_bboxes(
            item['img']/255,
            item['gt_bboxes'],
            item['gt_labels'] - 1,
            class_names=dataset.CLASSES,
            show=True,
            # show=not args.not_show,
            out_file=filename,
            wait_time=args.show_interval)
        # progress_bar.update()
        # print(item['gt_labels'] - 1)
        # print(item['filename'])

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
              ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
              ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
              ['bottle', ]]


def select_novel(novel_id):
    # novel_id = novel_id-1
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    print(cfg)
    dataset = build_dataset(cfg.data.train)
    novel_class_id = [CLASSES.index(c) for c in novel_sets[novel_id-1]]
    with open('mytest/novel_split%d_test.txt' % novel_id, 'w+') as f:
        for i in tqdm.tqdm(range(len(dataset))):
            item = dataset[i]
            filename = os.path.join(args.output_dir,
                                    Path(item['filename']).name
                                    ) if args.output_dir is not None else None
            for gt_l in (item['gt_labels'] -1):
                if gt_l in novel_class_id:
                    filename = item['filename'].split('/')[-1].split('.')[0]
                    f.write(filename)
                    f.write('\n')
                    break


if __name__ == '__main__':
    main1()
    # select_novel(1)

import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

from mmdet.datasets.builder import build_dataset

import tqdm
import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # or whatever other backend that you want
import matplotlib.colors
import matplotlib.pyplot as plt

from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET

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

def get_gt_num_per_class():
    cfg = retrieve_data_cfg('configs/few_shot/voc/voc_test.py', ['DefaultFormatBundle', 'Normalize', 'Collect'])
    dataset = build_dataset(cfg.data.train)

    num_gt_per_class = dict()
    for c in CLASSES:
        num_gt_per_class[c] = 0

    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        gt_bbox = item['gt_bboxes']
        gt_labels = (item['gt_labels'] - 1)
        for gl, gb in zip(gt_labels, gt_bbox):
            cls_name = CLASSES[gl]
            num_gt_per_class[cls_name] += 1

    for key in num_gt_per_class.keys():
        print(key, ': ', num_gt_per_class[key])

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

def crop_gt():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    print(cfg)
    dataset = build_dataset(cfg.data.train)
    save_root = 'mytest/voc_instances/trainval/'
    # progress_bar = mmcv.ProgressBar(len(dataset))
    num_instances_per_cls = [0 for i in range(len(CLASSES))]
    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        # print(Path(item['filename']).name)
        img = item['img'].astype(np.uint8)
        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels'] - 1,
        gt_labels = gt_labels[0]

        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            x1, y1, x2, y2 = gt_bbox.astype(np.int)
            gt_instance = img[y1:y2+1, x1:x2+1]
            gt_label = int(gt_label)
            save_path = save_root + CLASSES[gt_label] + '/' + str(num_instances_per_cls[gt_label]) + '.jpg'
            num_instances_per_cls[gt_label] += 1
            cv2.imwrite(save_path, gt_instance)

    for i in range(len(num_instances_per_cls)):
        print('{}: {}'.format(CLASSES[i], num_instances_per_cls[i]))
            # cv2.imshow('1', gt_instance)
            # cv2.waitKey(0)
        #     print(img)
        # cv2.imshow('1', img)
        # cv2.waitKey(0)
        # cv2.imshow('1', item['img']/255)
        # cv2.waitKey(2000)
        # mmcv.imshow_det_bboxes(
        #     item['img']/255,
        #     item['gt_bboxes'],
        #     item['gt_labels'] - 1,
        #     class_names=dataset.CLASSES,
        #     show=True,
        #     # show=not args.not_show,
        #     out_file=filename,
        #     wait_time=args.show_interval)
        # progress_bar.update()
        # print(item['gt_labels'] - 1)
        # print(item['filename'])

def summary_gt_bbox():
    # args = parse_args()

    cfg = retrieve_data_cfg('configs/few_shot/voc/voc_test.py', ['DefaultFormatBundle', 'Normalize', 'Collect'])
    dataset = build_dataset(cfg.data.train)

    size_scales = [32, 64, 128, 256, 512, 1024]
    area_scales = [_*_ for _ in size_scales]
    def get_area_ind(area):
        for i in range(len(area_scales)):
            if area <= area_scales[i]:
                return i
        return len(area_scales)

    gt_bbox_areas = dict()
    for c in CLASSES:
        gt_bbox_areas[c] = np.zeros(len(area_scales) + 1, dtype=np.int)

    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        gt_bbox = item['gt_bboxes']
        gt_labels = (item['gt_labels'] - 1)
        for gl, gb in zip(gt_labels, gt_bbox):
            cls_name = CLASSES[gl]
            area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            ind = get_area_ind(area)
            gt_bbox_areas[cls_name][ind] += 1
    return gt_bbox_areas

def plot_bar(data, xlable=None):
    x = ['0-32', '32-64', '64-128', '128-256', '256-512', '512-1024', '1024+']
    plt.bar(range(len(x)), data)
    plt.xticks(range(len(x)), x)
    if xlable is not None:
        plt.xlabel(xlable)

def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index*bar_span, y, bar_width)
    plt.ylabel('Scores')
    plt.title('multi datasets')
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels)
    plt.legend(CLASSES)
    plt.show()

import pickle
def save_dict(d, f_name):
    f_save = open(f_name, 'wb')
    pickle.dump(d, f_save)
    f_save.close()

def load_dict(f_name):
    # # 读取
    f_read = open(f_name, 'rb')
    dict2 = pickle.load(f_read)
    return dict2

def gt_num_from_xml():
    with PathManager.open(
            os.path.join('data/VOCdevkit/VOC2007', "ImageSets", "Main", 'test' + ".txt")
    ) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    num_gt_per_class = dict()
    for c in CLASSES:
        num_gt_per_class[c] = 0

    for fileid in fileids:
        anno_file = os.path.join('data/VOCdevkit/VOC2007', "Annotations", fileid + ".xml")
        jpeg_file = os.path.join('data/VOCdevkit/VOC2007', "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not (cls in CLASSES):
                continue
            num_gt_per_class[cls] += 1

    for key in num_gt_per_class.keys():
        print(key, ': ', num_gt_per_class[key])

if __name__ == '__main__':
    # main1()
    crop_gt()
    # select_novel(1)
    # get_gt_num_per_class()
    # gt_num_from_xml()

    # f_name = 'mytest/gt_bbox_areas_voc_test07_1333_800.pkl'
    # if os.path.exists(f_name):
    #     gt_bbox_areas = load_dict(f_name)
    # else:
    #     gt_bbox_areas = summary_gt_bbox()
    #     save_dict(gt_bbox_areas, f_name)
    # l = []
    # for c in novel_sets[1]:
    #     l.append(gt_bbox_areas[c])
    # datas = np.stack(l, axis=0)
    # labels = ['0-32', '32-64', '64-128', '128-256', '256-512', '512-1024', '1024+']
    # print(datas.shape)
    # create_multi_bars(labels, datas)
    # plt.show()

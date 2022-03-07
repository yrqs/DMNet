import os

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')
yolodir = '/home/luyue/Documents/few-shot-object-detection'
for seed in range(30):
    for shot in [10, 5, 3, 2, 1]:
        ids = {
            'VOC2007': [],
            'VOC2012': []
        }
        for c in CLASSES:
            with open(yolodir + '/datasets/vocsplit/seed%d/box_%dshot_%s_train.txt'%(seed, shot, c)) as f:
                content = f.readlines()
            for c in content:
                c_split = c.strip().split('/')
                ids[c_split[1]].append(c_split[-1][:-4])
        for key in ids.keys():
            ids[key] = list(set(ids[key]))
        # ids = list(set(ids))
        with open('mytest/few_shot_split/voc/VOC2007/ImageSets/Main/seed%d/trainval_%dshot_novel_standard.txt'%(seed, shot), 'w+') as f:
            for i in ids['VOC2007']:
                if '_' not in i:
                    f.write(i + '\n')
        with open('mytest/few_shot_split/voc/VOC2012/ImageSets/Main/seed%d/trainval_%dshot_novel_standard.txt'%(seed, shot), 'w+') as f:
            for i in ids['VOC2012']:
                if '_' in i:
                    f.write(i + '\n')

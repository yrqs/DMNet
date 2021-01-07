import os
import shutil
import tqdm

data_root = 'data/VOCdevkit/'

ann_files = [
    data_root + 'VOC2007/ImageSets/Main/trainval_3shot_novel_standard.txt',
    data_root + 'VOC2012/ImageSets/Main/trainval_3shot_novel_standard.txt'
]

img_prefixs = [data_root + 'VOC2007/', data_root + 'VOC2012/']

save_root = './mytest/temp'

for ann_file, img_prefix in zip(ann_files, img_prefixs):
    save_dir = save_root + ('/VOC2007' if '2007' in img_prefix else '/VOC2012')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            image = img_prefix + 'JPEGImages/' + line + '.jpg'
            shutil.copy(image, save_dir)


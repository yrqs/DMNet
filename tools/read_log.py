import os
import numpy as np
import argparse

work_dir_base = 'work_dirs/ga_retina_dml4_voc_split2/wo_norm/'
sub_dirs = {
    'voc': ['base', '1shot', '2shot', '3shot', '5shot', '10shot'],
    'coco': ['base', '10shot' '30shot']
}

dataset = ['voc', 'coco'][0]

def parse_args():
    parser = argparse.ArgumentParser(description='read log file')
    parser.add_argument('--dir_base', default='none', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def get_highest_voc_mAP(log_path):
    mAP_list = []
    novel1_list = []
    novel2_list = []
    novel3_list = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('| mAP         |'):
                mAP_list.append(float(line[-8:-2]))
            if '| novel_set1  | mAP' in line:
                novel1_list.append(float(line[-8:-2]))
            if '| novel_set2  | mAP' in line:
                novel2_list.append(float(line[-8:-2]))
            if '| novel_set3  | mAP' in line:
                novel3_list.append(float(line[-8:-2]))
    mAP_max = np.max(np.array(mAP_list))
    novel1_max = np.max(np.array(novel1_list))
    novel2_max = np.max(np.array(novel2_list))
    novel3_max = np.max(np.array(novel3_list))
    # idx = 2
    # mAP_max = mAP_list[idx]
    # novel1_max = novel1_list[idx]
    # novel2_max = novel2_list[idx]
    # novel3_max = novel3_list[idx]
    return mAP_max, novel1_max, novel2_max, novel3_max

def summury_voc_finetune(dir_base):
    results = dict()
    for sub_dir in sub_dirs[dataset][1:]:
        work_dir = os.path.join(dir_base, sub_dir)
        files = os.listdir(work_dir)
        log_path = None
        for file in files:
            if file.endswith('.log'):
                log_path = os.path.join(work_dir, file)
                break
        assert log_path is not None
        mAP_max, novel1_max, novel2_max, novel3_max = get_highest_voc_mAP(log_path)
        results[sub_dir] = 'mAP: ' + '%.3f'%mAP_max + \
                           '\tnovel1: ' + '%.3f'%novel1_max + \
                           '\tnovel2: ' + '%.3f'%novel2_max + \
                           '\tnovel3: ' + '%.3f'%novel3_max

    for key in results.keys():
        print('{}\t {}'.format(key, results[key]))

def summury_voc_finetuneG(dir_base, seed_range):
    results = dict()
    if not isinstance(seed_range, tuple):
        seed_range = (0, seed_range)
    for sub_dir in sub_dirs[dataset][1:2]:
        mAP_max_list = []
        novel1_max_list = []
        novel2_max_list = []
        novel3_max_list = []
        for seed in range(*seed_range):
            work_dir = os.path.join(dir_base, sub_dir, 'seed%d'%seed)
            files = os.listdir(work_dir)
            log_path = None
            for file in files:
                if file.endswith('.log'):
                    log_path = os.path.join(work_dir, file)
                    break
            assert log_path is not None
            mAP_max, novel1_max, novel2_max, novel3_max = get_highest_voc_mAP(log_path)
            mAP_max_list.append(mAP_max)
            novel1_max_list.append(novel1_max)
            novel2_max_list.append(novel2_max)
            novel3_max_list.append(novel3_max)
        print(novel1_max_list)
        mAP_max_mean = np.mean(np.array(mAP_max_list))
        novel1_max_mean = np.mean(np.array(novel1_max_list))
        novel2_max_mean = np.mean(np.array(novel2_max_list))
        novel3_max_mean = np.mean(np.array(novel3_max_list))
        results[sub_dir] = 'mAP: ' + '%.3f'%mAP_max_mean + \
                           '\tnovel1: ' + '%.3f'%novel1_max_mean + \
                           '\tnovel2: ' + '%.3f'%novel2_max_mean + \
                           '\tnovel3: ' + '%.3f'%novel3_max_mean

    for key in results.keys():
        print('{}\t {}'.format(key, results[key]))

if __name__ == '__main__':
    args = parse_args()
    if args.dir_base is not 'none':
        dir_base = args.dir_base
    else:
        dir_base = work_dir_base
    summury_voc_finetune(dir_base)
    # summury_voc_finetuneG(dir_base, 5)

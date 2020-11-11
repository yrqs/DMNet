import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from mmdet.core import voc_classes

from glob import glob
from tqdm import tqdm
from PIL import Image

coco_cls20_name = [
        'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
        'potted plant', 'sheep', 'couch', 'train', 'tv',
    ]
coco_cls40_name = ['truck', 'skateboard', 'banana', 'stop sign', 'parking meter',
                   'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                   'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                   'potted plant', 'sheep', 'couch', 'train', 'tv', ]
cocobase_cls20_name = ['truck', 'skateboard', 'banana', 'stop sign', 'parking meter',
                   'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                    ]
coco_cls40_20_name = ['truck1', 'skateboard1', 'banana1', 'stop sign1', 'parking meter1',
                   'bench1', 'elephant1', 'bear1', 'zebra1', 'giraffe1', 'backpack1', 'umbrella1',
                   'handbag1', 'tie1', 'suitcase1', 'frisbee1', 'skis1', 'snowboard1', 'sports ball1', 'kite1',
                   'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                   'potted plant', 'sheep', 'couch', 'train', 'tv', ]
coco_cls80_name = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                       'potted plant', 'sheep', 'couch', 'train', 'tv',
                       'truck', 'traffic light', 'fire hydrant', 'stop sign',
                       'parking meter', 'bench',
                       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                       'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                       'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                       'bed', 'toilet',
                       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# start at 21
cocobase_cls80_60_name = [' 0 ' ,' 1 ' ,' 2 ' ,' 3 ' ,' 4 ' ,' 5 ' ,' 6 ' ,' 7 ' ,' 8 ' ,' 9 ' ,' 10 ' ,' 11 ' ,' 12 ' ,' 13 ' ,' 14 ' ,' 15 ' ,' 16 ' ,' 17 ' ,' 18 ' ,' 19 ',
                        'truck', 'traffic light', 'fire hydrant', 'stop sign',
                       'parking meter', 'bench',
                       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                       'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                       'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                       'bed', 'toilet',
                       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush']

used_names = cocobase_cls80_60_name

label_ids = {name: i + 1 for i, name in enumerate(used_names)}
print(len(cocobase_cls80_60_name))


def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in used_names:
            continue
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w*h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": category_id,
                        "id": anno_id,
                        "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = []
    for k,v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():
    xml_path = 'data/coco_xml/val2017/Annotations'
    img_path = 'data/coco_xml/val2017/JPEGImages'
    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, 'data/tmp_out/cocobase_cls80_60.json')
    print('Done!')


if __name__ == '__main__':
    main()

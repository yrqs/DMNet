import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

CLASSES_VOC = ('plane', 'bike', 'bird*', 'boat', 'bottle', 'bus*', 'car', 'cat',
               'chair', 'cow*', 'table', 'dog', 'horse', 'motorbike*', 'person',
               'pottedplant', 'sheep', 'sofa*', 'train', 'tv')

CLASSES_COCO = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CLASSES = CLASSES_VOC

novel_sets = [['bird', 'bus', 'cow', 'motorbike', 'sofa'],
              ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
              ['boat', 'cat', 'motorbike', 'sheep', 'sofa']]


def show_feature(feature, use_sigmoid=True, bar_scope=None):
    norm = None
    if bar_scope:
        vmin, vmax = bar_scope
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure()
    if use_sigmoid:
        feature = feature.sigmoid()
    im = plt.imshow(feature.clone()[0][0].cpu(), cmap='rainbow', norm=norm)
    fig.colorbar(im)
    plt.show()

def show_dis(dis, bar_scope=None):
    norm = None
    if bar_scope:
        vmin, vmax = bar_scope
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure()
    im = plt.imshow(dis.clone().cpu(), cmap='rainbow', norm=norm)
    fig.colorbar(im)

    num_cls = len(CLASSES)
    scale_ls = range(num_cls)
    label_ls = list(CLASSES)
    plt.yticks(scale_ls, label_ls)

    plt.show()
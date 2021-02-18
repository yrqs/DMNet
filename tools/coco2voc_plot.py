# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
del matplotlib.font_manager.weight_dict['roman']
# for key in matplotlib.font_manager.weight_dict:
#     print(key)
# matplotlib.font_manager._rebuild()
plt.rc('font', family='Times New Roman')

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
x = np.array(['YOLO-ft-full', 'FSRW', 'FRCN-ft-full$^1$', 'Meta\nR-CNN', 'MetaDet', 'FRCN-ft-full$^\dag$', 'MPSR', 'DMNet\n(Ours)', 'DMNet$^\dag$\n(Ours)'])  # x值取默认值
y = np.array([28.29, 32.29, 31.2, 37.4, 34, 39.3, 42.3, 43.6, 44.3])

front_size = 30
text_x_offset = 0.0
# 定义函数来显示柱状上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.25 + text_x_offset, 1.005 * height, '%.2f' % (height), fontsize=front_size)

plt.rc('axes', axisbelow=True)
plt.grid(b=True, axis='y', alpha=0.5, linestyle='-.') #只显示y轴网格线
plt.xticks(np.arange(len(x)), x, fontsize=front_size)
a = plt.bar(np.arange(len(x)), y, color=['g', 'b', 'c', 'm', 'chocolate', 'deepskyblue', 'darkorchid', 'gold', 'r'], width=0.5)
autolabel(a)
plt.ylim(ymin=26, ymax=45.5)
plt.ylabel('mAP (%)', fontsize=front_size)
plt.tick_params(labelsize=front_size)
plt.show()
# author：lph
# funtion:image enhance

import cv2
import numpy as np
from tkinter import *
from skimage import filters, exposure
import matplotlib.pyplot as plt
from skimage.morphology import disk
from matplotlib.font_manager import FontProperties

# 读入图片
im = cv2.imread('./image/image.png', 0)
im_copy = cv2.imread('./image/image.png', 0)
# 如果图片为空，返回错误信息，并终止程序
if im is None:
    print("图片打开失败！")
    exit()
# 中值滤波去噪
medStep = 3  # 设置为3*3的滤波器


def m_filter(x, y, step):
    """中值滤波函数"""
    sum_s = []  # 定义空数组
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(im[x + k][y + m])  # 把模块的像素添加到空数组
    sum_s.sort()  # 对模板的像素由小到大进行排序
    return sum_s[(int(step * step / 2) + 1)]


for i in range(int(medStep / 2), im.shape[0] - int(medStep / 2)):
    for j in range(int(medStep / 2), im.shape[1] - int(medStep / 2)):
        im_copy[i][j] = m_filter(i, j, medStep)  # 用模板的中值来替代原像素的值
cv2.imshow("Median", im_copy)
# Gamma校正
img3 = exposure.adjust_gamma(im_copy, 1.05)
cv2.imshow("Gamma", img3)


# 对比度、亮度增强
def Contrast_and_Brightness(alpha, beta, img):
    """使用公式f(x)=α.g(x)+β"""
    # α调节对比度，β调节亮度
    blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
    return dst


img4 = Contrast_and_Brightness(1.1, 30, img3)
cv2.imshow("Contrast", img4)
# 创建一个窗口
plt.figure('对比图', figsize=(7, 5))
# 中文字体设置
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 新宋体
# 显示原图
plt.subplot(121)  # 子图1
# 显示原图，设置标题和字体
plt.imshow(im, plt.cm.gray), plt.title('处理前图片', fontproperties=font)

# 显示处理过的图像
plt.subplot(122)  # 子图2
# 显示处理后的图，设置标题和字体
plt.imshow(img4, plt.cm.gray), plt.title('处理后图片', fontproperties=font)
plt.show()
# 销毁所有窗口
cv2.destroyAllWindows()
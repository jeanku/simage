import cv2
import os
from PIL import ImageFont, ImageDraw, Image as Pimage
import numpy as np
from matplotlib import pyplot as plt
#
filepath = './Resource/timg1.jpg'
savepath = './Resource/test4.jpg'
# # open
#
# img = cv2.imread(filepath, 1)
img = np.zeros((512, 512, 3), np.uint8)
img.fill(255)
#
# # 展示
# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # 划线
# cv2.line(img, (0,0), (256, 256), (0, 255, 255), 4)
cv2.rectangle(img, (256, 0), (512, 256), (0, 255, 0), 3)
# cv2.circle(img, (256, 256), 200, (0, 255, 255, 0.1), -1)
# # cv2.circle(img, (256 + 56, 256 - 98), 50, (0, 0, 255, 0.2), -1)
# # cv2.ellipse(img, (256, 128), (200, 50), 0, 30, 360, 255, -1)
# # cv2.ellipse(img, (256, 128), (200, 50), 0, 30, 360, 255, -1)
#
# # pts = np.array([[50, 0], [150, 0], [150, 200], [50, 200]], np.int32)
# # pts = pts.reshape((-1, 1, 2))
# # cv2.polylines(img, [pts], True, (0, 255, 255), 5)
#
#
# # 添加文字
# # font = cv2.FONT_HERSHEY_SIMPLEX
# # cv2.putText(img, 'OpenCV', (10, 500), font, 4, (0, 255, 255), 2, cv2.LINE_AA)
#
# # numpy 处理
# # img[0:512:50, ::5] = (0, 255, 0)
# # temp = img[0:200, 982:1182]
# # img[465:665, 0:200] = img[465:665, 0:200] * 0.8 + temp * 0.2
# # img[465:665, 0:200] = cv2.addWeighted(img[465:665, 0:200], 0.8, temp, 0.2, 0)
#
#
# # img[:,:,0] = 255 - img[:,:,0] * 0.8
# # img[:,:,1] = 255 - img[:,:,1] * 0.1
# # img[:,:,2] = 255 - img[:,:,2] * 0.1
#
# # 图片加法
# # img = np.zeros((512, 512, 3), np.uint8)
# # img.fill(128)
# # img1 = np.zeros((512, 512, 3), np.uint8)
# # img1.fill(255)
# # img = img + img1                # numpy 加法取余数
# # # img = cv2.add(img, img1)      # cv2 加法最大255
#
#
# # img = np.zeros((512, 512, 3), np.uint8)
# # img.fill(0)
# # cv2.putText(img, "OPENCV", (10, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 245), 10)
#
# img = cv2.imread(filepath, 1)
#
# # img = np.zeros((512, 512, 3), np.uint8)
# # img.fill(0)
# #
# # img[0:100, 0:100] = (0, 255, 0)
# # img.fill((0, 255, 0))
# # img = np.zeros((512, 512, 3), np.uint8)
# # img.fill(0)
# # cv2.rectangle(img, (0, 0), (256, 256), (255, 255, 255), -1)
#
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #
# # low = np.array([80, 20, 100])
# # high = np.array([110, 250, 255])
# # mask = cv2.inRange(hsv, low, high)
#
#
# # print(img[0, 0])
# # print(mask[850, 630])
# # print(mask[0, 0])
# # mask_rev = cv2.bitwise_not(mask)

# # print(img[0, 0])
# # bg_img = cv2.bitwise_or(img, (255, 255, 255), mask=mask)
# # # print(bg_img[0,0])
# # # # exit(0)
# # img = cv2.bitwise_and(img, img, mask=mask_rev)
# # print(img[0, 0])
# # # exit(0)
# #
# #
# # img = cv2.addWeighted(bg_img, 1, img, 1, 0)
# # print(img[0, 0])
#
# # print()
#
# # print(img.shape)
# # exit(0)
#
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread(filepath, 0)
# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'MEAN', 'GAUSSIAN']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, th2, th3]
#
#     for i in range(8):
#         plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
#
#     # plt.show()
#
# # 保存
cv2.imwrite(savepath, img)
#
# # plt
# # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# # plt.show()
#
# # print(Test.name)

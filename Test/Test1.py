import numpy as np
import cv2
from Util.Image import Image
from Util.PltImage import PltImage

# img = np.zeros((3, 3), dtype=np.uint8)
# img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# filepath = '../Resource/timg1.jpg'
# img2 = Image().open(filepath, 0)
# # img3 = img2.GaussianBlur((15, 15), 0)
#
# # g_hpf = img2.image - img3.image
# # Image().init(g_hpf).show()
#
# kernel_33 = np.array([
#     [-1, -1, -1],
#     [-1, 8, -1],
#     [-1, -1, -1],
# ])
# kernel_55 = np.array([
#     [-1, -1, -1, -1, -1],
#     [-1, 1, 2, 1, -1],
#     [-1, 2, 4, 2, -1],
#     [-1, 1, 2, 1, -1],
#     [-1, -1, -1, -1, -1],
# ])
# from scipy import ndimage
#
# k3 = ndimage.convolve(img2.image, kernel_33)
# k5 = ndimage.convolve(img2.image, kernel_55)
#
# blurred = cv2.GaussianBlur(img2.image, (15, 15), 0)
# g_hpf = img2.image - blurred
# cv2.imshow('33', k3)
# cv2.imshow('55', k5)
# cv2.imshow('g_hpf', blurred)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # print(img1.item(2, 2, 0))
# # img1.itemset((2, 2, 0), 129)
# # print(img1.item(2, 2, 0))
# # Image().init(img1).show()
# exit(0)

# filepath = '../Resource/timg1.jpg'
# fileresult = '../Resource/result.jpg'
#
# from Util.Filter.VConvolutionFilter import *
#
# img = cv2.imread(filepath, 0)
# img2 = SharpenFilter().apply(img)
# img3 = FindEdgesFilter().apply(img)
# img4 = BlurFilter().apply(img)
# img5 = EmbossFilter().apply(img)
#
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
# cv2.imshow('img4', img4)
# cv2.imshow('img5', img5)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Image().open(filepath).canny(100, 100).show()
# Image().blank((200, 200), 0)
import numpy

# img = np.zeros((3, 3), dtype=numpy.uint8)
# img[0:2, 0:2] = 128
# img = Image().blank((400, 400), 0)
# img.image[100:300, 100:300] = 255
# ret, thresh = cv2.threshold(img.image, 127, 255, 0)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# color = cv2.cvtColor(img.image, cv2.COLOR_GRAY2BGR)
# img1 = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
# Image().init(ret).show()


# img = Image().open(fileresult)
# fileresult = '../Resource/logo.jpg'
#
# img = cv2.imread(fileresult)
# img[:] = 255 - img
#
# # print(img)
# # exit(0)
# # imgray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# #
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # #
# # print(contours)
# # exit(0)
#
# # 绘制独立轮廓，如第四个轮廓：
# imgres = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# # # 但是大多数时候，下面的方法更有用：
# # img = cv2.drawContours(img.image, contours, 3, (0, 255, 0), 3)
#
# cv2.imshow('img5', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Image().init(img2).show()


'''


fileresult = '../Resource/logo1.jpg'
oimg = Image().open(fileresult).image
img = Image().open(fileresult).grayImage().threshold(127, 255, cv2.THRESH_BINARY).image
img[:] = 255 - img

_, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
Image().open(fileresult).contours_hull(contours, (0, 255, 255)).show()
# Image().open(fileresult).contours_ellipse(contours, (0, 255, 255)).show()
# Image().open(fileresult).contours_hull(contours, color=(0, 255, 255)).show()
#
# exit(0)

cnt = contours[0]

epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
hull = cv2.convexHull(cnt)

oimg = cv2.drawContours(oimg, hull, -1, (0, 0, 255), 2)


# curr = hull[0]
# for i in hull:
#     cv2.line(oimg, (curr[0][0], curr[0][1]), (i[0][0], i[0][1]), (0, 0, 255), 3)
#     curr = i
#
# cv2.draw
#
# # oimg1 = cv2.drawContours(Image().open(fileresult).image, contours1, -1, (0, 0, 255), 2)
# # oimg2 = cv2.drawContours(Image().open(fileresult).image, contours2, -1, (0, 0, 255), 2)
# # oimg3 = cv2.drawContours(Image().open(fileresult).image, contours3, -1, (0, 0, 255), 2)
# # oimg4 = cv2.drawContours(Image().open(fileresult).image, contours4, -1, (0, 0, 255), 2)
# # oimg5 = cv2.drawContours(Image().open(fileresult).image, contours5, -1, (0, 0, 255), 2)
# #
# #
# cv2.imshow('oimg1', oimg1)
# # cv2.imshow('oimg2', oimg2)
# # cv2.imshow('oim3', oimg3)
# # cv2.imshow('oim4', oimg4)
#
#
#
# # for c in contours1:
# #     # x, y, w, h = cv2.boundingRect(c)
# #     # cv2.rectangle(oimg, (x,y), (x+w, y+h), (0, 255, 0), 3)
# #     #
# #     # rect = cv2.minAreaRect(c)
# #     #
# #     # box = cv2.boxPoints(rect)
# #     #
# #     # box = np.int0(box)
# #     #
# #     # cv2.drawContours(oimg, [box], 0, (0, 0, 255), 2)
# #
# #     (x, y), radius = cv2.minEnclosingCircle(c)
# #     center = (int(x), int(y))
# #     radius = int(radius)
# #     cv2.circle(oimg, center, radius, (0, 255, 0), 2)
#
# # cv2.drawContours(oimg, hull, -1, (255, 0, 0), 2)
cv2.imshow("ha", oimg)
# # # print(thresh[0, 0])
# # # exit(0)
# # #
# # img2 = cv2.pyrDown(img)
#
# cv2.imshow('img', thresh)
# cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()

'''

'''
img = Image().open('../Resource/logo1.jpg')
img.image[:] = 255 - img.image
ret, thresh = cv2.threshold(img.grayImage(), 127, 255, cv2.THRESH_BINARY)
_, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#image 1
# cv2.drawContours(img.image, contours, -1, (0, 0, 255), 2)
# img.show()
#image 2
# img.contours_hull(contours).show()
'''

'''
#极坐标
img = Image().open('../Resource/ditu.jpg')
img.image[:] = 255 - img.image
gray = img.grayImage()
ret, thresh = cv2.threshold(gray, 0, 50, cv2.THRESH_BINARY)

_, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in contours:
    if i.__len__() > 100:
        cnt = i
        mask = np.zeros(gray.shape, np.uint8)
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        img.line(leftmost, leftmost, (0, 255, 255), 3)
        img.line(rightmost, rightmost, (0, 255, 255), 3)
        img.line(topmost, topmost, (0, 0, 255), 3)
        img.line(bottommost, bottommost, (255, 0, 255), 3)
        img.show()

exit(0)

# 极坐标2
img = Image().blank((4, 4), 0)
img.fill_ploy([[1, 0], [2, 0], [3, 1], [3, 2], [2, 3], [1, 3], [0, 2], [0, 1]])
gray = img.grayImage()
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

img.line(leftmost, leftmost, (0, 255, 255), 1)
img.line(rightmost, rightmost, (0, 255, 255), 1)
img.line(topmost, topmost, (0, 0, 255), 1)
img.line(bottommost, bottommost, (255, 0, 255), 1)
img.show('512313')
exit(0)


img = Image().blank((15, 15), 0)
img.fill_ploy([[0, 7], [6, 6], [7, 0], [8, 6], [14, 7], [8, 8], [7, 14], [6, 8]])
gray = img.grayImage()
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img.image, start, end, [0, 255, 0], 2)
    cv2.circle(img.image, far, 5, [0, 0, 255], -1)

img.show()

print(contours)

img.contours_hull(contours, thickness=1).show()
print(contours)
exit(0)

cnt = contours[0]

hull = cv2.convexHull(cnt)

print(hull)
exit(0)

defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, [0, 255, 0], 2)
    cv2.circle(img, far, 5, [0, 0, 255], -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''''
# import numpy as np
# from matplotlib import pyplot as plt
# filepath = '../Resource/timg1.jpg'
# img = cv2.imread(filepath)
#
# mask = np.zeros(img.shape[:2], np.uint8)
#
# bgdModdel = np.zeros((1, 65), np.floor(64))
# fgdModdel = np.zeros((1, 65), np.floor(64))
#
# rect = (100, 50, 421, 378)
# cv2.grabCut(img, mask, rect, bgdModdel, fgdModdel, 5, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
#
# plt.subplot(121), plt.imshow(img)
# plt.title('grabcut'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY))
#
# plt.title('original'), plt.xticks([]), plt.yticks([])
# plt.show()


import os

path = '../Resource/ditu.jpg'
# dd = os.walk(path)
count = 13
img = Image().open(path)
# img.image[img.image >= 255] = [1, 1, 1]
# cv2.bitwise_not(img.image, img.image)

img.resize(400, 400).save('../Resource/TestSet/' + '{}.png'.format(count))

# count = 1
# for dirname, dirnames, filenames in dd:
#     for i in filenames:
#         if os
#         Image().open(path + i).resize(400 ,400).save(path + '{}.png' . format(count))
#         count += 1
# exit(0)


# import cv2
# import numpy as np
# import sys
#
#
# filepath = '../Resource/result.jpg'
# img = Image().open(filepath)
# gray = img.grayImage()
#
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptor = sift.detectAndCompute(gray, None)
#
# cv2.drawKeypoints(image=img.image, outImage=img.image, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 0, 236))
# img.show('hahah')

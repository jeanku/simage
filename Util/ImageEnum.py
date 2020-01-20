import cv2
import os
from PIL import ImageFont, ImageDraw, Image as Pimage
import numpy as np
from Util.MagicMetaClass import MagicMetaClass


class Image(MagicMetaClass):
    pass

if __name__ == '__main__':
    filepath = './Resource/test.jpg'
    savepath = './Resource/test4.jpg'


    # img = Image.imread(filepath, Image.THRESH_BINARY)
    # print(type(cv2.imread(filepath)) == np.ndarray)
    # print(Image.imread(filepath))
    # print(Image.THRESH_BINARY)
    # print(Image.THRESH_OTSU)
    # print(Image.imread())
    # exit(0)

    '''
    # # 打开某一个图片 并保存
    Image.open(filepath).save(savepath)

    # # 创建空图 并保存
    Image.blank((100, 50), 0).save(savepath)

    # 打开某一个图片 并展示
    Image.open(filepath).show()

    # 添加文字
    Image.open(filepath).text('健康评级:', (87, 310), 38, 0, fontpath).save(savepath)

    # 图片大小变化
    Image.open(filepath).resize(512, 512).save(savepath)

    # 图片按比例缩放
    Image.open(filepath).scale(0.5).save(savepath)

    # 图片位移
    Image.open(filepath).warpaffine(100, 0, 0).save(savepath)

    # 图片旋转
    Image.open(filepath).rotation((50, 0), -45, 1, 0).save(savepath)

    # 图片仿射变换
    img = Image.open(filepath)
    points1 = [img.point(0, 0), img.point(1, 0), img.point(0, 1)]
    points2 = [img.point(1, 0), img.point(0, 0), img.point(1, 1)]
    img.transform2(points1, points2).save(savepath)

    # 图片水平翻转
    Image.open(filepath).hflip().save(savepath)

    # 图片垂直翻转
    Image.open(filepath).vflip().save(savepath)

    # 图片透视变换
    img = Image.open(filepath)
    points1 = [[0, 0], [img.maxX, 0], [0, img.maxY], [img.maxX, img.maxY]]
    points2 = [[img.midX * 0.5, 50], [img.midX * 1.5, 50], [0, img.maxY], [img.midX, img.midY]]
    img.transform3(points1, points2).save(savepath)

    # 蒙版添加logo(会除去logo背景色）
    logo = Image.open(logopath)
    Image.open(filepath).paste_mask(logo.image, (20, 20)).save(savepath)

    # 正常添加logo
    logo = Image.open(logopath)
    Image.open(filepath).paste(logo.image, (20, 20)).save(savepath)

    # 按长&宽比例获取坐标
    point = Image.open(filepath).point(0.5, 0.5)            # 获取图片中心点坐标
    point = Image.open(filepath).point()                    # 获取图片大小
    print(point)

    # 负坐标换算
    point = Image.open(filepath).point_neg(20, 20)
    
    # img1 = Image.open(filepath).image
    # img2 = Image.open(filepath).equalize_hist_color().image
    # img3 = Image.open(filepath).equalize_hist_color_byblock((5,5)).equalize_hist_color_byblock((12, 12)).image
    # # img3 = Image.open(filepath, 0).equalize_hist_gray().image
    # img4 = Image.open(filepath).equalize_hist_gray_byblock().equalize_hist_gray_byblock().image
    # Image().pltshow([img1, img2, img3, img4])
    '''

    # import cv2
    # import numpy as np
    # from matplotlib import pyplot as plt
    # 
    # img = cv2.imread(filepath)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 
    # plt.imshow(hist, interpolation='nearest')
    # plt.show()


    img = cv2.rectangle(Image.blank((512, 512), 0).image, (128, 128), (394, 394), (255, 255, 255), -1)
    Image.init(img).save(savepath)
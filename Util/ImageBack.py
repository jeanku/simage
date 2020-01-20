import cv2
import os
from PIL import ImageFont, ImageDraw, Image as Pimage
import numpy as np
from Util.MagicMetaClass import MagicMetaClass


class Image():

    def __getattr__(self, item):
        self.__image__ = getattr(cv2, item)(self.__image__)
        return self

    __image__ = None

    IMAGE_TYPE_JPG = 'jpg'
    IMAGE_TYPE_PNG = 'png'

    THRESH_BINARY = cv2.THRESH_BINARY
    THRESH_BINARY_INV = cv2.THRESH_BINARY_INV
    THRESH_TRUNC = cv2.THRESH_TRUNC
    THRESH_TOZERO = cv2.THRESH_TOZERO
    THRESH_TOZERO_INV = cv2.THRESH_TOZERO_INV
    THRESH_OTSU = cv2.THRESH_OTSU  # Otsu 二值化

    ADPTIVE_THRESH_MEAN_C = cv2.ADAPTIVE_THRESH_MEAN_C  # 阈值取自相邻区域的平均值
    ADPTIVE_THRESH_GAUSSIAN_C = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 阈值取值相邻区域的加权和，权重为一个高斯窗口。

    @staticmethod
    def open(path, flags=1):
        '''
        读取图片
        :param path: 文件路径
        :param flags:
            -1：imread按解码得到的方式读入图像
             0：imread按单通道的方式读入图像，即灰白图像
             1：imread按三通道方式读入图像，即彩色图像
        :return:
        '''
        instance = Image()
        instance.__image__ = cv2.imread(path, flags)
        return instance

    @staticmethod
    def blank(size, bgcolor=255):
        '''
        创建空白图
        :param size: 大小（width, height）
        :param bgcolor: 颜色 255:白 0:黑
        :return:
        '''
        if not (type(size) == tuple and size.__len__() == 2):
            raise Exception('invalid size')
        instance = Image()
        instance.__image__ = np.zeros((size[1], size[0], 3), np.uint8)
        instance.__image__.fill(bgcolor)
        return instance

    @staticmethod
    def init(img):
        instance = Image()
        instance.__image__ = img
        return instance

    def show(self, title='Image'):
        '''
        图片展示
        :param title:title
        :return:
        '''
        cv2.imshow(title, self.__image__)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def close(self):
        '''
        close
        :return:
        '''
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save(self, name, quality=50):
        '''
        文件保存
        :param name: 保存的文件路径名
        :param quality: 保存的图片质量 0-100
        :return:
        '''
        type = self._file_extension(name)
        if type == self.IMAGE_TYPE_JPG:
            cv2.imwrite(name, self.__image__, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if type == self.IMAGE_TYPE_PNG:
            cv2.imwrite(name, self.__image__, [int(cv2.IMWRITE_PNG_COMPRESSION), int(quality / 10)])
        cv2.waitKey()
        cv2.destroyAllWindows()

    def tobytes(self, type='.png'):
        '''
        图片转成二进制流
        :param type:
        :return:
        '''
        return cv2.imencode(type, self.image)[1].tobytes()

    def _file_extension(self, name):
        '''
        获取文件后缀名
        :param name:文件名称
        :return: string
        '''
        return os.path.splitext(name)[-1][1:]

    def to_gray(self):
        if len(self.__image__.shape) == 3:
            self.__image__ = self.__image__ = cv2.cvtColor(self.__image__, cv2.COLOR_BGR2GRAY)
        return self

    def show_calchist_gray(self):
        '''
        灰度直方图展示
        :return:null
        '''
        import matplotlib.pyplot as plt
        plt.plot(cv2.calcHist([self.to_gray().image], [0], None, [256], [0, 256]), color='g')
        plt.show()

    def show_calchist_color(self):
        '''
        颜色直方图
        :return:null
        '''
        import matplotlib.pyplot as plt
        chans = cv2.split(self.__image__)
        colors = ('b', 'g', 'r')
        plt.figure()
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.show()

    def equalize_hist_gray(self):
        '''
        灰度直方图均衡
        :return:null
        '''
        cv2.equalizeHist(self.to_gray().image, self.__image__)
        return self

    def equalize_hist_color(self):
        '''
        彩色直方图均衡
        :return:
        '''
        ycrcb = cv2.cvtColor(self.__image__, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, self.__image__)
        return self

    def equalize_hist_gray_byblock(self, size=(8, 8)):
        '''
        CLAHE彩色直方图均衡
        :return:
        '''
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=size)
        self.__image__ = clahe.apply(self.to_gray().image)
        return self

    def equalize_hist_color_byblock(self, size=(8, 8)):
        '''
        CLAHE彩色直方图均衡
        :return:
        '''
        b, g, r = cv2.split(self.__image__)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=size)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        self.__image__ = cv2.merge([b, g, r])
        return self

    def pltshow(self, images):
        import matplotlib.pyplot as plt
        i = 221
        for index in images:
            shape = len(index.shape)
            if shape == 3:
                plt.subplot(i), plt.imshow(index[:, :, ::-1])
            elif shape == 2:
                plt.subplot(i), plt.imshow(index, cmap=plt.cm.gray)
            i += 1
        plt.show()

    def font(self, path):
        pass

    @property
    def image(self):
        return self.__image__

    @property
    def maxX(self):
        return self.shape[1]

    @property
    def midX(self):
        return int(self.shape[1] * 0.5)

    @property
    def maxY(self):
        return self.shape[0]

    @property
    def midY(self):
        return int(self.shape[0] * 0.5)

    @property
    def height(self):
        '''
        获取图片的height
        :return: int
        '''
        return self.shape[0]

    @property
    def width(self):
        '''
        获取图片的width
        :return: int
        '''
        return self.shape[1]

    @property
    def shape(self):
        '''
        获取图片的shape
        :return: (rows, cols)
        '''
        return self.__image__.shape[:2]


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
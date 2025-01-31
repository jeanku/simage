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

    def text(self, content, point, size, color, fontpath):
        '''
        添加文本
        :param content: string 文本内容
        :param point: 起始点坐标(0, 0)
        :param size: 大小
        :param color: 字体颜色 : (100, 200, 200) （B，G, R, A)
        :param fontpath: 字体路径
        :return: 
        '''
        font = ImageFont.truetype(fontpath, size)
        img_pil = Pimage.fromarray(self.__image__)
        draw = ImageDraw.Draw(img_pil)
        draw.text(point, content, font=font, fill=color)
        self.__image__ = np.array(img_pil)
        return self

    def resize(self, width, height):
        '''
        几何变换: 扩展缩放
        :param width: 变化后宽度
        :param height: 变化后高度
        :return: self
        '''
        if width >= self.width and height >= self.height:  # 拓展
            type = cv2.INTER_CUBIC
        else:  # 缩放
            type = cv2.INTER_AREA
        self.__image__ = cv2.resize(self.__image__, (width, height), interpolation=type)
        return self

    def scale(self, ratio):
        '''
        几何变换: 图片等比例缩放
        :param ratio: 缩放比例 0-1:缩小  > 1: 放大
        :return: self
        '''
        self.resize(int(self.width * ratio), int(self.height * ratio))
        return self

    def warpaffine(self, x, y, bgcolor=0):
        '''
        几何变换: 图片平移
        :param x: 沿x轴方向移动x像素(支持正负)
        :param y: 沿y轴方向移动y像素(支持正负)
        :param bgcolor: 背景填充色
        :return: self
        '''
        mov = np.float32([[1, 0, x], [0, 1, y]])
        self.__image__ = cv2.warpAffine(self.__image__, mov, (self.width, self.height), borderValue=bgcolor)
        return self

    def rotation(self, point, angle, ration, bgcolor=0):
        '''
        几何变换: 图片旋转
        :param point: 旋转中心(x, y)
        :param angle: 旋转角度
        :param ration: 旋转后缩放因子
        :param bgcolor: 背景填充色
        :return:
        '''
        mov = cv2.getRotationMatrix2D(point, angle, ration)
        self.__image__ = cv2.warpAffine(self.__image__, mov, (2 * point[0], 2 * point[1]), borderValue=bgcolor)
        return self

    def transform2(self, points1, points2):
        '''
        几何变换: 仿射变换 (二维空间变化）
        :param points1: list 变化前页面三个点
        :param points2: list 变化后页面三个点
        :return: self
        '''
        mov = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
        self.__image__ = cv2.warpAffine(self.__image__, mov, self.point())
        return self

    def hflip(self):
        '''
        水平翻转
        :return: self
        '''
        pts1 = np.float32([[0, 0], [self.maxX, 0], [0, self.maxY]])
        pts2 = np.float32([[self.maxX, 0], [0, 0], [self.maxX, self.maxY]])
        return self.transform2(pts1, pts2)

    def vflip(self):
        '''
        垂直翻转
        :return: self
        '''
        pts1 = np.float32([[0, 0], [self.maxX, 0], [0, self.maxY]])
        pts2 = np.float32([[0, self.maxY], [self.maxX, self.maxY], [0, 0]])
        return self.transform2(pts1, pts2)

    def transform3(self, points1, points2):
        '''
        几何变换: 透视变换(3维空间变化）
        :param points1: 变化前页面4个点
        :param points2: 变化后页面4个点
        :return: self
        '''
        mov = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))
        self.__image__ = cv2.warpPerspective(self.__image__, mov, self.point())
        return self

    def paste_mask(self, img, point):
        '''
        通过mask添加图片
        :param img: logo cv2对象
        :param point: logo位置(x, y)
        :return: self
        '''
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)  # 二值化函数
        mask_inv = cv2.bitwise_not(mask)

        height, width = img.shape[:2]
        if point[0] >= 0:
            r1 = point[0]
            r2 = r1 + width
        else:
            r1 = self.width - width + point[0]
            r2 = self.width + point[0]

        if point[1] >= 0:
            c1 = point[1]
            c2 = c1 + height
        else:
            c1 = self.height - height + point[1]
            c2 = self.height + point[1]
        roi = self.__image__[c1:c2, r1:r2]
        # 4，ROI和Logo图像融合
        img_bg = cv2.bitwise_and(roi, roi, mask=mask)
        logo_fg = cv2.bitwise_and(img, img, mask=mask_inv)

        dst = cv2.add(img_bg, logo_fg)
        self.__image__[c1:c2, r1:r2] = dst
        return self

    def paste(self, img, point):
        '''
        添加图片
        :param img: logo cv2对象
        :param point: logo位置(top, left)
        :return: self
        '''
        height, width = img.shape[:2]
        inst_width = self.width
        inst_height = self.height
        if abs(point[0]) > inst_width or abs(point[1]) > inst_height:
            raise Exception('坐标有误')
        if point[0] >= 0:
            r1 = point[0]
            r2 = min(r1 + width, inst_width)
        else:
            r1 = inst_width - width + point[0]
            r2 = min(inst_width + point[0], inst_width)

        if point[1] >= 0:
            c1 = point[1]
            c2 = min(c1 + height, inst_height)
        else:
            c1 = inst_height - height + point[1]
            c2 = min(inst_height + point[1], inst_height)
        self.__image__[c1:c2, r1:r2] = img
        return self

    def point(self, x_ratio=1, y_ratio=1):
        '''
        根据比例获取坐标点
        :param x_ratio: x轴比例
        :param y_ratio: y轴比例
        :return: (x, y)
        '''
        return int(self.width * x_ratio), int(self.height * y_ratio)

    def point_neg(self, x, y):
        '''
        负坐标转化绝对坐标
        :param point: (x, y)
        :return: (x, y)
        '''
        width, height = self.point()
        return x if x >= 0 else x + width, y if y >= 0 else y + height

    def threshold(self, value, repvalue, type):
        '''
        简单阀值
        :param value: 阈值
        :param repvalue: 替换值
        :param type: 类型:
                • Image.THRESH_BINARY
                • Image.THRESH_BINARY_INV
                • Image.THRESH_TRUNC
                • Image.THRESH_TOZERO
                • Image.THRESH_TOZERO_INV
                • Image.THRESH_OTSU                   #  Otsu 二值化
        :return:
        '''
        ret, self.__image__ = cv2.threshold(self.__image__, value, repvalue, type)
        return self

    def threshold_adaptive(self, method, block_size=11, const=2):
        '''
        自适应阀值
        :param method:
              – Image.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值
              – Image.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域的加权和，权重为一个高斯窗口。
        :param block_size: 11
        :param const: 2
        :return:
        '''
        gray = cv2.cvtColor(self.__image__, cv2.COLOR_RGB2GRAY)
        gauss = cv2.GaussianBlur(gray, (3, 3), 1)
        self.__image__ = cv2.adaptiveThreshold(gauss, 255, method, cv2.THRESH_BINARY, block_size, const)
        return self

    def filter2D(self, kernel):
        '''
        2D 卷积
        :param kernel: np.ones((5, 5), np.float32) / 25
        :return: self
        '''
        self.__image__ = cv2.filter2D(self.__image__, -1, kernel)
        return self

    def blur(self, kernel):
        '''
        平均 卷积框覆盖区域所有像素的平均值来代替中心元素
        :param kernel: (5, 5)
        :return: self
        '''
        self.__image__ = cv2.blur(self.__image__, kernel)
        return self

    def gaussian_blur(self, *args, **kw):
        '''
        高斯模糊
        :param kernel: (5, 5)
        :return: self
        '''
        self.__image__ = cv2.GaussianBlur(self.__image__, *args, **kw)
        return self

    def median_blur(self, *args, **kw):
        '''
        中值模糊
        :param kernel: cv2.median_blur(img,5)
        :return: self
        '''
        self.__image__ = cv2.medianBlur(self.__image__, *args, **kw)
        return self

    def bilateral_filter(self, *args, **kw):
        '''
        双边滤波
        :param kernel: cv2.bilateralFilter(img,9,75,75)
        :return: self
        '''
        self.__image__ = cv2.bilateralFilter(self.__image__, *args, **kw)
        return self

    def erode(self, *args, **kw):
        '''
        腐蚀
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.erode(self.__image__, *args, **kw)
        return self

    def dilate(self, *args, **kw):
        '''
        膨胀
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.dilate(self.__image__, *args, **kw)
        return self

    def morphologyEx(self, *args, **kw):
        '''
        开运算
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.morphologyEx(self.__image__, *args, **kw)
        return self

    def canny(self, *args, **kw):
        '''
        边缘检测
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.Canny(self.__image__, *args, **kw)
        return self

    def pyrup(self, *args, **kw):
        '''
        图像金字塔 up
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.pyrUp(self.__image__, *args, **kw)
        return self

    def pyrdown(self, *args, **kw):
        '''
        图像金字塔 up
        :param kernel: np.ones((5,5),np.uint8)
        :return: self
        '''
        self.__image__ = cv2.pyrDown(self.__image__, *args, **kw)
        return self

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
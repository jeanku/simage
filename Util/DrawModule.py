import cv2
import numpy as np
from Util.BaseImage import BaseImage
from Util.MagicMetaClass import MagicMetaClass


class DrawModule(BaseImage, MagicMetaClass):

    def line(self, *args, **kw):
        self.__image__ = cv2.line(self.__image__, *args, **kw)
        return self

    def rectangle(self, *args, **kw):
        self.__image__ = cv2.rectangle(self.__image__, *args, **kw)
        return self

    def circle(self, *args, **kw):
        self.__image__ = cv2.circle(self.__image__, *args, **kw)
        return self

    def ellipse(self, *args, **kw):
        self.__image__ = cv2.ellipse(self.__image__, *args, **kw)
        return self

    def putText(self, *args, **kw):
        self.__image__ = cv2.putText(self.__image__, *args, **kw)
        return self

    def grayImage(self):
        return cv2.cvtColor(self.__image__, cv2.COLOR_BGR2GRAY)

    def text(self, content, point, size, color, fontpath):
        from PIL import ImageFont, ImageDraw, Image as Pimage
        '''
        添加文本(可以添加中文)
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

    def warpAffine(self, x, y, bgcolor=0):
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
        return cv2.threshold(self.__image__, value, repvalue, type)

    def adaptiveThreshold(self, method, block_size=11, const=2):
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
        self.__image__ = cv2.filter2D(self.__image__, -1, kernel)
        return self

    def blur(self, kernel):
        self.__image__ = cv2.blur(self.__image__, kernel)
        return self

    def GaussianBlur(self, *args, **kw):
        self.__image__ = cv2.GaussianBlur(self.__image__, *args, **kw)
        return self

    def medianBlur(self, *args, **kw):
        self.__image__ = cv2.medianBlur(self.__image__, *args, **kw)
        return self

    def bilateralFilter(self, *args, **kw):
        self.__image__ = cv2.bilateralFilter(self.__image__, *args, **kw)
        return self

    def erode(self, *args, **kw):
        self.__image__ = cv2.erode(self.__image__, *args, **kw)
        return self

    def dilate(self, *args, **kw):
        self.__image__ = cv2.dilate(self.__image__, *args, **kw)
        return self

    def morphologyEx(self, *args, **kw):
        self.__image__ = cv2.morphologyEx(self.__image__, *args, **kw)
        return self

    def canny(self, *args, **kw):
        self.__image__ = cv2.Canny(self.__image__, *args, **kw)
        return self

    def pyrup(self, *args, **kw):
        self.__image__ = cv2.pyrUp(self.__image__, *args, **kw)
        return self

    def pyrdown(self, *args, **kw):
        self.__image__ = cv2.pyrDown(self.__image__, *args, **kw)
        return self

    def contours_rectangle(self, contours, color=(0, 255, 0), thickness=3):
        """轮廓矩形区域"""
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.__image__, (x, y), (x + w, y + h), color, thickness)
        return self

    def contours_min_rectangle(self, contours, color=(0, 255, 0), thickness=3):
        """轮廓最小矩形区域"""
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.__image__, [box], 0, color, thickness)
        return self

    def contours_circle(self, contours, color=(0, 255, 0), thickness=3):
        """轮廓最小闭圆"""
        for c in contours:
            (x, y), radius = cv2.minEnclosingCircle(c)
            cv2.circle(self.__image__, (int(x), int(y)), int(radius), color, thickness)
        return self

    def contours_ellipse(self, contours, color=(0, 255, 0), thickness=3):
        """轮廓最小椭圆"""
        for c in contours:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(self.__image__, ellipse, color, thickness)
        return self

    def contours_hull(self, contours, rate=0.1, color=(0, 255, 0), thickness=3):
        for cnt in contours:
            # epsilon = rate * cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(cnt)
            length = len(hull)
            for i in range(length):
                cv2.line(self.__image__, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), color, thickness)
        return self



    def fill_ploy(self, points, color=(255, 255, 255)):
        """根据点绘制填充图形"""
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(self.__image__, [pts], color)
        return self


if __name__ == '__main__':
    filepath = '../Resource/test.jpg'
    savepath = '../Resource/test4.jpg'
    # DrawModule().blank((512, 512), 255).line((256, 0), (510, 256), (0, 0, 255), 3).save(savepath)
    # DrawModule().blank((512, 512), 255).rectangle((256, 0), (510, 256), (0, 0, 255), 3).save(savepath)
    # DrawModule().blank((512, 512), 255).circle((256, 256), 200, (0, 255, 255, 0.1), -1).save(savepath)
    # DrawModule().blank((512, 512), 255).ellipse((256, 128), (200, 50), 0, 30, 360, 255, -1).save(savepath)
    # DrawModule().blank((512, 512), 255).putText("OPENCV", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 245), 10).save(savepath)

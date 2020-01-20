import cv2
import numpy as np
import os


class BaseImage():
    __image__ = None

    IMAGE_TYPE_JPG = 'jpg'
    IMAGE_TYPE_PNG = 'png'

    def open(self, path, flags=1):
        '''
        读取图片
        :param path: 文件路径
        :param flags:
            -1：imread按解码得到的方式读入图像
             0：imread按单通道的方式读入图像，即灰白图像
             1：imread按三通道方式读入图像，即彩色图像
        :return:
        '''
        self.__image__ = cv2.imread(path, flags)
        return self

    def blank(self, size, bgcolor=255):
        '''
        创建空白图
        :param size: 大小（width, height）
        :param bgcolor: 颜色 255:白 0:黑
        :return:
        '''
        if not (type(size) == tuple and size.__len__() == 2):
            raise Exception('invalid size')
        self.__image__ = np.zeros((size[1], size[0], 3), np.uint8)
        self.__image__.fill(bgcolor)
        return self

    def init(self, img):
        self.__image__ = img
        return self

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

    def pltshow(self, images = None):
        from .PltImage import PltImage
        if images is None:
            PltImage.show([self.__image__])
        else:
            PltImage.show(images)

    def _file_extension(self, name):
        '''
        获取文件后缀名
        :param name:文件名称
        :return: string
        '''
        return os.path.splitext(name)[-1][1:]

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
    filepath = '../Resource/test.jpg'
    savepath = '../Resource/test4.jpg'
    BaseImage().open(filepath).show()


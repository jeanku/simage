import cv2
import numpy

'''滤波器'''


class VConvolutionFilter(object):
    """A filter that applies a convolutioin to V(or all of BGR) 卷积滤波器"""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst=None):
        """Apply the filter with a BGR or gray source/destination."""
        return cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius 锐化滤波器"""

    def __init__(self):
        VConvolutionFilter.__init__(self, numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius 边缘检测滤波器"""

    def __init__(self):
        VConvolutionFilter.__init__(self, numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))


class BlurFilter(VConvolutionFilter):
    """a blur filter with a 2-pixel radius.临近平均滤波器"""

    def __init__(self):
        kernel = numpy.array([
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
        ])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An enboss filter wiyj a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ])
        VConvolutionFilter.__init__(self, kernel)

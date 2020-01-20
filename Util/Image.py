import cv2
import os
from PIL import ImageFont, ImageDraw, Image as Pimage
import numpy as np
from Util.BaseImage import BaseImage
from Util.DrawModule import DrawModule


class Image(DrawModule):
    pass


if __name__ == '__main__':

    filepath = '../Resource/test.jpg'
    savepath = '../Resource/test4.jpg'


    Image().open(filepath).pltshow()
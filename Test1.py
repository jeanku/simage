# import cv2
# import os
# from PIL import ImageFont, ImageDraw, Image as Pimage
# import numpy as np
# from matplotlib import pyplot as plt
#
from Image import Image
import random

filepath = './Resource/timg1.jpg'
savepath = './Resource/result.jpg'

img = Image.blank((10, 10), 0).image
print(random.randint(0, 255))
# exit(0)
img[:, :] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
for i in img:
    for j in i:
        j = [0, 255, 128]

Image.init(img).save(savepath)
print(img)
exit(0)
exit(0)
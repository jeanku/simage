class PltImage():

    @staticmethod
    def showImages(images, titles=[], type=None):
        import matplotlib.pyplot as plt
        import math
        length = len(images)
        if length < 4:
            pass
        else:
            rows = math.ceil(8 / 4)
            for i in range(length):
                plt.subplot(rows, 4, i + 1), plt.imshow(images[i], type)
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def show(images):
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

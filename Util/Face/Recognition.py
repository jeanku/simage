import face_recognition
from Util.Image import Image


class Recognition():
    __regimage__ = None
    __repath__ = None

    def __init__(self, filepath):
        self.__repath__ = filepath
        self.__regimage__ = face_recognition.load_image_file(filepath)

    def image(self):
        '''
        获取图片对象
        :param filepath: 文件路径
        :return: object
        '''
        return self.__regimage__

    def locate(self):
        '''
        面部检测
        :return: 每个face的(top, right, bottom, left)坐标  [(top, right, bottom, left), ...]
        '''
        return face_recognition.face_locations(self.__regimage__)

    def landmarks(self):
        '''
        面部各个轮廓识别
        :param filepath:
        :return: [
            {
                'chin': [(x, y), (x, y)],
                'left_eyebrow': [(x, y), (x, y)],
                'right_eyebrow': [(x, y), (x, y)],
                'nose_bridge': [(x, y), (x, y)],
                'nose_tip': [(x, y), (x, y)],
                'left_eye': [(x, y), (x, y)],
                'right_eye': [(x, y), (x, y)],
                'top_lip': [(x, y), (x, y)],
                'bottom_lip': [(x, y), (x, y)],
            }
        ]
        '''
        return face_recognition.face_landmarks(self.__regimage__)

    def check(self, file):
        '''
        人脸识别
        :param file:
        :return: true: 和init的图片属于用一个人   false: 和init的图片不属于用一个人
        '''
        image = face_recognition.load_image_file(file)
        biden_encoding = face_recognition.face_encodings(self.__regimage__)[0]
        unknown_encoding = face_recognition.face_encodings(image)[0]
        return face_recognition.compare_faces([biden_encoding], unknown_encoding)


if __name__ == '__main__':
    filepath = '../../Test/Face/Image/u=60469401,60394384&fm=26&gp=0.jpg'
    unknowfilepath = '../../Test/Face/Image/u=520636890,2178500167&fm=26&gp=0.jpg'
    img = Image().open(filepath)

    '''
     # 脸检测
    re = Recognition(filepath).locate()
    for t, r, b, l in re:
        img.rectangle((l, t), (r, b), (0, 0, 255), 2)
    img.show('gaga')
    '''

    '''
    # 面部检测
    re = Recognition(filepath).landmarks()

    for key, val in re[0].items():
        if key == 'chin':
            for x, y in val:
                img.circle((x, y), 1, (0, 255, 255), 1)

        if key == 'left_eyebrow' or key == 'right_eyebrow':
            for x, y in val:
                img.circle((x, y), 1, (0, 0, 0), 1)

        if key == 'nose_bridge' or key == 'nose_tip':
            for x, y in val:
                img.circle((x, y), 1, (0, 255, 0), 1)

        if key == 'left_eye' or key == 'right_eye':
            for x, y in val:
                img.circle((x, y), 1, (255, 255, 0), 1)

        if key == 'top_lip' or key == 'bottom_lip':
            for x, y in val:
                img.circle((x, y), 1, (0, 0, 255), 2)
    img.show('haha')
    '''

    '''
    import os
    path = '../../Test/Face/Image/'
    dd = os.walk(path)
    mol = Recognition(filepath)

    for dirname, dirnames, filenames in dd:
        for i in filenames:
            res = mol.check('../../Test/Face/Image/' + i)
            print(res)
    exit(0)
    '''
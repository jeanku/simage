import face_recognition
# from PIL import Image
from Util.Image import Image


def get_image(filepath):
    '''
    获取图片对象
    :param filepath: 文件路径
    :return: object
    '''
    return face_recognition.load_image_file(filepath)


def face_locate(image):
    return face_recognition.face_locations(image)

def landmarks(image):
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
            'right_eye': [(x, y), (x, y)],
            'right_eye': [(x, y), (x, y)],
            'top_lip': [(x, y), (x, y)],
            'bottom_lip': [(x, y), (x, y)],
        }
    ]
    '''
    return face_recognition.face_landmarks(image)


# filepath = './Image/u=60469401,60394384&fm=26&gp=0.jpg'
filepath = '../../Resource/timg2.jpeg'
img = get_image(filepath)
faces = face_locate(img)

cvimg = Image().open(filepath)
for t, r, b, l in faces:
    face_image = img[t:b, l:r]
    landmks = landmarks(face_image)
    if landmks:
        cvimg.rectangle((l, t), (r, b), (0, 0, 255), 2)
    else:
        print(landmks)

cvimg.show('haha')
# res = find_face('./Image/u=60469401,60394384&fm=26&gp=0.jpg')
# res = face_locate('../../Resource/timg2.jpeg')


# print(res)
exit(0)
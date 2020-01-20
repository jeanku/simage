import cv2
import os

filename = ''
key = 0
def face_detect(filename):
    global key
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (2, 2))
        if eyes.__len__() > 0:
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite('./TestImage/%s.pgm' % key, f)
            key = key + 1


path = './Image'

dd = os.walk(path)

for dirname, dirnames, filenames in dd:
    for i in filenames:
        face_detect('./Image/' + i)

# face_detect('../../Resource/timg2.jpeg')

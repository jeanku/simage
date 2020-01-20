import cv2
filename = ''

def face_detect(filename):
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
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# face_detect('../../Resource/timg2.jpeg')

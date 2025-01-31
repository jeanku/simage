import cv2
import numpy as np

filepath = '../Resource/dd.jpg'
savepath = '../Resource/cornerHarris.jpg'

# MEGTHOD1
# cornerHarris(一般角点检测)
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imwrite(savepath, img)

exit(0)


# MEGTHOD2
# cornerSubPix (模糊角点检测）
# img = cv2.imread(filepath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
# ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
# dst = np.uint8(dst)
# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
# # Now draw them
# res = np.hstack((centroids, corners))
# res = np.int0(res)
# img[res[:, 1], res[:, 0]] = [0, 0, 255]
# img[res[:, 3], res[:, 2]] = [0, 255, 0]
# cv2.imwrite('../Resource/cornerSubPix.jpg', img)


# MEGTHOD3
# goodFeaturesToTrack (优化角点检测）
# img = cv2.imread(filepath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
# corners = np.int0(corners)
#
# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(img, (x, y), 3, 255, -1)
# cv2.imwrite('../Resource/goodFeaturesToTrack.jpg', img)


# MEGTHOD4
# img = cv2.imread(filepath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# img = cv2.drawKeypoints(img, kp, gray) -
# cv2.imwrite('../Resource/drawKeypoints.jpg', img)

# METHOD5
# img = cv2.imread(filepath)
# surf = cv2.xfeatures2d.SURF_create(5000)
# kp, des = surf.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 2)

# cv2.imwrite('../Resource/drawKeypoints_2.jpg', img)


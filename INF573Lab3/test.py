import numpy as np
import cv2 as cv

img1 = cv.imread('IMG_0045.JPG')
img2 = cv.imread('IMG_0046.JPG')
# Q5. does it work when the images are rotated so that they are not approximately aligned at first ?
# img1 = cv.imread('IMG_0045.JPG')
# img2 = cv.imread('IMG_0046r.JPG')
# Q6. make it work on 2 images of your own
# img1 = cv.imread('WechatIMG19.JPG')
# img2 = cv.imread('WechatIMG20.JPG')

img1gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Q1. use an opencv feature extractor and descriptor to detect and compute features on both images
sift = cv.xfeatures2d_SIFT().create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1gray, None)
kp2, des2 = sift.detectAndCompute(img2gray, None)
kp1_image = cv.drawKeypoints(img1gray, kp1, None)
kp2_image = cv.drawKeypoints(img2gray, kp2, None)
cv.imshow("kp1", kp1_image)
cv.imshow("kp2", kp2_image)
cv.waitKey(0)

# Q2. use a descriptor matcher, to compute feature correspondences
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
# ratio test as per Lowe's paper
for m, n in matches:
	if m.distance < 0.7 * n.distance:
		good.append(m)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
cv.imshow("good", img3)
cv.waitKey(0)

# Q3. Organize the matched feature pairs into vectors and estimate an homography using RANSAC
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
K, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 3.0)

# Q4. copy I1 to a new (bigger image) K using the identity homography
# warp I2 to K using the computed homography
warpImg = cv.warpPerspective(img2, K, (img1.shape[1] + img2.shape[1], img2.shape[0]))
warpImg[0:img1.shape[0], 0:img1.shape[1]] = img1
cv.imshow("warpImg", warpImg)
cv.waitKey(0)



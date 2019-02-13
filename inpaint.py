import cv2
import numpy as np
from skimage.restoration import inpaint


original_img = cv2.imread('/home/sonya/zoomos/images/632963.jpg')

template = cv2.imread('/home/sonya/Downloads/632963_mask.jpg', 0)

ret, mask = cv2.threshold(template, 254, 255, cv2.THRESH_BINARY_INV)

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

result_img = original_img.copy()

for i in range(10, 254, 2):

	img = result_img.copy()

	sharp_img = cv2.filter2D(img, -1, kernel_sharpening)

	mask_with_img = cv2.bitwise_and(sharp_img, sharp_img, mask = mask)

	mask_with_img = cv2.cvtColor(mask_with_img, cv2.COLOR_RGB2GRAY)

	ret, mask_with_img = cv2.threshold(mask_with_img, i, 255, cv2.THRESH_BINARY)

	new, contours_light, hierarchy = cv2.findContours(mask_with_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	new_mask = cv2.drawContours(img, contours_light, -1, (0, 0, 0), 2)

	new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)

	ret, new_mask = cv2.threshold(new_mask, 1, 255, cv2.THRESH_BINARY_INV)

	result_img = cv2.inpaint(result_img, new_mask, 7, cv2.INPAINT_NS)

result_img = cv2.inpaint(result_img, new_mask, 7, cv2.INPAINT_NS)

cv2.imshow('result_img', result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

original_img = cv2.imread('/home/sonya/zoomos/images/577986.jpg', 0)
template = cv2.imread('template.jpg', 0)

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

sharp_img = cv2.filter2D(original_img, -1, kernel_sharpening)
sharp_template = cv2.filter2D(template, -1, kernel_sharpening)

w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

mask = np.zeros(sharp_img.shape, sharp_img.dtype)

for meth in methods:
    method = eval(meth)

    res = cv2.matchTemplate(sharp_img, sharp_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(original_img, top_left, bottom_right, (255, 0, 255), cv2.FILLED)

# result_img = cv2.inpaint(original_img, mask, 10, cv2.INPAINT_NS)

cv2.imshow('original', original_img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
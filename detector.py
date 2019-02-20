import cv2
import numpy as np
import os

color = cv2.COLOR_BGR2GRAY

kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

images_dir = "/home/sonya/zoomos/watermark-removal/images/";

results_dir = images_dir + "results/"
items_dir = images_dir + "items/"
templates_dir = images_dir + "templates/"

for item_filename in os.listdir(items_dir):

	original_img = cv2.imread(items_dir + item_filename)
	img_color = cv2.cvtColor(original_img, color)
	sharp_img = cv2.filter2D(img_color, -1, kernel_sharpening)

	saved = False

	for template_filename in os.listdir(templates_dir):

		if saved:
			break

		template = cv2.imread(templates_dir + template_filename)
		template_color = cv2.cvtColor(template, color)

		for i in np.arange(2.0, 0.5, -0.03):
			resized_template = cv2.resize(template_color, (0,0), fx=i, fy=i)
			sharp_template = cv2.filter2D(resized_template, -1, kernel_sharpening)

			w, h = sharp_template.shape[::-1]

			method = eval('cv2.TM_CCOEFF_NORMED')

			if w > sharp_img.shape[1] or h > sharp_img.shape[0]:
				continue

			res = cv2.matchTemplate(sharp_img, sharp_template, method)

			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			if max_val < 0.4:
				continue

			top_left = max_loc

			if top_left[1] == 0 or top_left[1] == sharp_img.shape[0]:
				continue

			bottom_right = (top_left[0] + w, top_left[1] + h)

			cv2.rectangle(original_img, top_left, bottom_right, (255, 0, 255), cv2.FILLED)

			cv2.imwrite(results_dir + item_filename, original_img)

			saved = True
			break

	if not saved:
		cv2.imwrite(results_dir + "bad/" + item_filename, original_img)
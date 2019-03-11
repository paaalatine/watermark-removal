import os

from commons import *


def detect(img):
	
	img_color = cv2.cvtColor(img, color)
	sharp_img = cv2.filter2D(img_color, -1, kernel_sharpening)

	for template_filename in os.listdir(templates_dir):

		template = cv2.imread(templates_dir + template_filename)
		template_color = cv2.cvtColor(template, color)

		for i in np.arange(3.0, 0.5, -0.03):
			resized_template = cv2.resize(template_color, (0,0), fx=i, fy=i)
			sharp_template = cv2.filter2D(resized_template, -1, kernel_sharpening)

			w, h = sharp_template.shape[::-1]

			method = eval('cv2.TM_CCOEFF_NORMED')

			if w > sharp_img.shape[1] or h > sharp_img.shape[0]:
				continue

			res = cv2.matchTemplate(sharp_img, sharp_template, method)

			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			if i >= 3 and max_val < tresholds[shop][0]:
				continue

			if i >= 2 and i < 3 and max_val < tresholds[shop][1]:
				continue

			if i < 2 and max_val < tresholds[shop][2]:
				continue

			top_left = max_loc

			if top_left[1] == 0 or top_left[1] == sharp_img.shape[0]:
				continue

			bottom_right = (top_left[0] + w, top_left[1] + h)
			
			mask = np.zeros((sharp_img.shape[0], sharp_img.shape[1], 3), np.uint8)
			
			return top_left, bottom_right

	print("undefined" + img)
	return 0, 0
	

def main():
	for item_filename in os.listdir(items_dir):
		img = cv2.imread(items_dir + item_filename)
		top_left, bottom_right = detect(img)
		if top_left == 0 and bottom_right == 0:
			cv2.imwrite(results_dir + "bad/" + item_filename, img)
		else:
			cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), cv2.FILLED)
			cv2.imwrite(results_dir + item_filename, img)
	

if __name__ == "__main__":
	main()
	
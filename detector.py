import os

from commons import *


def match(img, template):

	for i in np.arange(3.0, 0.5, -0.03):

		resized_template = cv2.resize(template, (0,0), fx=i, fy=i)
		sharp_template = cv2.filter2D(resized_template, -1, kernel_sharpening)

		h_template, w_template = sharp_template.shape
		h_img, w_img = img.shape

		if h_template > h_img or w_template > w_img:
			continue

		method = eval('cv2.TM_CCOEFF_NORMED')

		res = cv2.matchTemplate(img, sharp_template, method)

		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		if i >= 3 and max_val < tresholds[shop][0]:
			continue

		if i >= 2 and i < 3 and max_val < tresholds[shop][1]:
			continue

		if i < 2 and max_val < tresholds[shop][2]:
			continue

		top_left = max_loc

		bottom_right = (top_left[0] + w_template, top_left[1] + h_template)
		
		return top_left, bottom_right

	return 0, 0


def detect(img):
	
	img_gray = cv2.cvtColor(img, color)
	sharp_img = cv2.filter2D(img_gray, -1, kernel_sharpening)

	for template_filename in os.listdir(templates_dir):

		template = cv2.imread(templates_dir + template_filename)
		template_gray = cv2.cvtColor(template, color)

		top_left, bottom_right = match(sharp_img, template_gray)

		if top_left != 0 or bottom_right != 0:
			return top_left, bottom_right

		if shop == "onliner.by":
			h, w = template_gray.shape[:2]
			M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
			template_gray = np.rot90(template_gray, 1)
			# cv2.imshow('image',template_gray)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			top_left, bottom_right = match(sharp_img, template_gray)
			if top_left != 0 or bottom_right != 0:
				return top_left, bottom_right


	print("undefined")
	return 0, 0
	

def main():
	for item_filename in os.listdir(items_dir):
		print(item_filename + " detecting..")
		img = cv2.imread(items_dir + item_filename)
		top_left, bottom_right = detect(img)
		if top_left == 0 and bottom_right == 0:
			cv2.imwrite(results_dir + "bad/" + item_filename, img)
		else:
			cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), cv2.FILLED)
			cv2.imwrite(results_dir + item_filename, img)
	

if __name__ == "__main__":
	main()
	
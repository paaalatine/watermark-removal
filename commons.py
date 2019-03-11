import cv2
import numpy as np

color = cv2.COLOR_BGR2GRAY

shop = "onliner.by"
tresholds = {
                'pleer.ru': [0.48, 0.42, 0.31], 
                'onliner.by': [1.0, 0.8, 0.5]
            }

kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])

images_dir = "../images/";

results_dir = images_dir + "results/"
items_dir = images_dir + "items/" + shop + "/"
templates_dir = images_dir + "templates/" + shop + "/"
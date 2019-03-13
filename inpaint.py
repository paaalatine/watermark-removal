from detector import *


def inpaint(img):
    
    top_left, bottom_right = detect(img)
    
    if top_left == 0 and bottom_right == 0:
        return []
    
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), cv2.FILLED)
    
    for i in range(10, 254, 2):
        sharp_img = cv2.filter2D(img, -1, kernel_sharpening)
        
        mask_with_img = cv2.bitwise_and(sharp_img, sharp_img, mask = mask)
        mask_with_img = cv2.cvtColor(mask_with_img, cv2.COLOR_RGB2GRAY)
        
        ret, mask_with_img = cv2.threshold(mask_with_img, i, 255, cv2.THRESH_BINARY)
        
        new, contours_light, hierarchy = cv2.findContours(mask_with_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        new_mask = cv2.drawContours(img, contours_light, -1, (0, 0, 0), 2)
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
        
        ret, new_mask = cv2.threshold(new_mask, 1, 255, cv2.THRESH_BINARY_INV)
        
        img = cv2.inpaint(img, new_mask, 7, cv2.INPAINT_NS)
        
    img = cv2.inpaint(img, new_mask, 7, cv2.INPAINT_NS)
    
    return img


def main():
    for item_filename in os.listdir(items_dir):
        print(item_filename + " processing..")
        img = cv2.imread(items_dir + item_filename)
        inpainted_img = inpaint(img)     
        if inpainted_img.size == 0:
            cv2.imwrite(results_dir + "bad/" + item_filename, img)
        else:
            cv2.imwrite(results_dir + item_filename, inpainted_img)


if __name__ == "__main__":
    main()


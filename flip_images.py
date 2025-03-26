#after capturing gesture flip the images
import cv2, os

def flip_images():
	gest_folder = "gestures"
	images_labels = []#line initializes an empty list 
	images = []#line initializes an empty list 
	labels = []#line initializes an empty list 
	for g_id in os.listdir(gest_folder):
		for i in range(1200):
			path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+1+1200)+".jpg"
			print(path)
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)#line flips the image horizontally using the cv2.flip() function
			cv2.imwrite(new_path, img)

flip_images()

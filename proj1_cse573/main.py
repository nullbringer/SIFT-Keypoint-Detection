import cv2
import numpy as np
from math import sqrt
from math import exp



def print_image(img, image_name):
	cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
	cv2.imshow(image_name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

def write_image(img, image_name):
	cv2.imwrite(image_name + '.png',img)





def convolve_img(img, kernel,kernel_radius):


	height, width = img.shape

	output_image = np.zeros(img.shape, np.uint8)

	# ignoring edge pixels for now.
	# add padding zero

	for i in range(kernel_radius, height-kernel_radius):
		for j in range(kernel_radius, width-kernel_radius):
			
			# output_image[i][j] = 	img[i-1][j-1] * kernel[0][0] +	\
			# 						img[i][j-1] * kernel[1][0] +	\
			# 						img[i+1][j-1] * kernel[2][0] + 	\
			# 						img[i-1][j] * kernel[0][1] + 	\
			# 						img[i][j] * kernel[1][1] + 		\
			# 						img[i+1][j] * kernel[2][1] + 	\
			# 						img[i-1][j+1] * kernel[0][2] + 	\
			# 						img[i][j+1] * kernel[1][2] + 	\
			# 						img[i+1][j+1] * kernel[2][2]





			# elementwise multiplication sum

			
			loop_end = (kernel_radius*2)+1

			sum = 0
			for x in range(0,loop_end):
				for y in range(0,loop_end):
					sum += kernel[x][y] * img[i-kernel_radius+x][j-kernel_radius+y]


			


			output_image[i][j] = sum



	return output_image



def edge_detection_x(img):

	x_kernel = np.array([				
							[-1, 0, 1], 
							[-2, 0, 2], 
							[-1, 0 , 1] 
						], np.int8)  

	edge_x_img = convolve_img(img,x_kernel,1)
	print_image(edge_x_img,'edge_x_img')



def edge_detection_y(img):

	y_kernel = np.array([				
							[-1, -2, -1], 
							[0, 0, 0], 
							[1, 2 , 1] 
						], np.int8)  


	edge_y_img = convolve_img(img,y_kernel,1)
	print_image(edge_y_img,'edge_y_img')


def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )


def get_gaussian_kernel(sigma):

	kernel_radius = 3

	# compute the actual kernel elements
	hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
	vkernel = [x for x in hkernel]
	kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]


	# normalize the kernel elements
	kernelsum = sum([sum(row) for row in kernel2d])
	kernel2d = [[x/kernelsum for x in row] for row in kernel2d]

	return kernel2d


def resize_image_to_half(img):

	height, width = img.shape

	output_image = np.zeros((int(height/2), int(width/2)), np.uint8)


	i_op = 0
	for i in range(0,height):
		j_op = 0

		if i%2 == 0:
				continue

		for j in range(0, width):
			if j%2 == 0:
				continue

			output_image[i_op][j_op] = img[i][j] 
			j_op+=1

		i_op+=1

	
	#print_image(output_image,'output_image')
	return output_image



def generate_gaussian_blur_for_an_image(img, octav_id, sigma_row):


	for i in range(len(sigma_row)):

		

		gussian_blurred_img = convolve_img(img, get_gaussian_kernel(sigma_row[i]), 3)
		#gussian_blurred_img  = cv2.GaussianBlur(img, (7,7), sigma_row[i],0)
		write_image(gussian_blurred_img,'gb_img_'+ octav_id +'_'+str(i))




def generate_octavs(image_1,sigma_table):
	
	# octav 1: original image 

	write_image(image_1,'octav_1_original')
	generate_gaussian_blur_for_an_image(image_1,'octav_1', sigma_table[0])



	# octav 2: original image/2
	image_2 = resize_image_to_half(image_1)
	write_image(image_2,'octav_2_original')
	generate_gaussian_blur_for_an_image(image_2,'octav_2', sigma_table[1])




	# octav 3: original image/4
	image_3 = resize_image_to_half(image_2)
	write_image(image_3,'octav_3_original')
	generate_gaussian_blur_for_an_image(image_3,'octav_3', sigma_table[2])


	# octav 4: original image/8
	image_4 = resize_image_to_half(image_3)
	write_image(image_4,'octav_4_original')
	generate_gaussian_blur_for_an_image(image_4,'octav_4', sigma_table[3])

	# print(sigma_table[3])


def compute_DoG():

	for i in range(0,4):
		for j in range(1,5):

			img_lower_blur = cv2.imread("gb_img_octav_" + str(j) + "_" + str(i) + ".png", 0)
			img_higher_blur = cv2.imread("gb_img_octav_" + str(j) + "_" + str(i+1) + ".png", 0)
			
			write_image(img_higher_blur-img_lower_blur,'dog_octav_'+ str(j)+'_'+ str(i))



def find_maxima_minima(octav_num,layer):

	output_image = cv2.imread("octav_" + str(octav_num) + "_original.png", 0)


	dog_top = cv2.imread("dog_octav_" + str(octav_num) + "_" + str(layer-1) + ".png", 0)
	dog_middle = cv2.imread("dog_octav_" + str(octav_num) + "_"+ str(layer) + ".png", 0)
	dog_bottom = cv2.imread("dog_octav_" + str(octav_num) + "_" + str(layer+1) + ".png", 0)



	height, width = dog_middle.shape


	# traversing image
	# ignoring edge pixels for now.
	# add padding zero

	for h in range(1,height-1):
		for w in range(1, width-1):

			#threshold
			if dog_middle[h][w]<200:
				continue


			# traversing and comparing 26 neighbours
			is_maxima = True

			for i in range(h-1,h+2):
				for j in range(w-1,w+2):
					if (dog_middle[h][w] < dog_middle[i][j]) or (dog_middle[h][w] < dog_top[i][j]) or (dog_middle[h][w] < dog_bottom[i][j]):
						is_maxima = False
						break

				if not is_maxima:
					break

			if is_maxima:
				output_image[h][w] = 255
			else:

				is_minima = False

				for i in range(h-1,h+2):
					for j in range(w-1,w+2):
						if (dog_middle[h][w] > dog_middle[i][j]) or (dog_middle[h][w] > dog_top[i][j]) or (dog_middle[h][w] > dog_bottom[i][j]):
							is_minima = False
							break

					if not is_minima:
						break
				if is_minima:
					output_image[h][w] = 255

	print_image(output_image,'keypoints'+str(layer)+str(h)+str(w))

def find_keypoints():



	#traversing 4 octavs
	for octav_num in range(1,5):
		

		# traversing 2 middle layers
		for layer in range(1,3):
			
			find_maxima_minima(octav_num,layer)
			


			



def main():

	#task 1

	task_1_img = cv2.imread("task1.png", 0)
	# edge_detection_x(task_1_img)
	# edge_detection_y(task_1_img)


	#task 2

	task_2_img = cv2.imread("task2.jpg", 0)
	# task_2_img = cv2.imread("testcat.jpg", 0)

	sigma_table = np.array([				
							[1/sqrt(2), 1, sqrt(2), 2, 2*sqrt(2)], 
							[sqrt(2), 2,  2*sqrt(2), 4, 4*sqrt(2)], 
							[2*sqrt(2), 4, 4*sqrt(2), 8, 8*sqrt(2)],
							[4*sqrt(2), 8, 8*sqrt(2), 16, 16*sqrt(2)]
						])


	

	# generate_octavs(task_2_img, sigma_table);
	# compute_DoG()
	find_keypoints()




	






main()
 















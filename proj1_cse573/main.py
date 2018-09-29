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



def convolve_img(img, kernel):
	height, width = img.shape

	output_image = np.zeros(img.shape, np.uint8)

	# igniring edge pixels for now.
	# add padding zero

	for i in range(1,height-1):
		for j in range(1, width-1):
			
			output_image[i][j] = 	img[i-1][j-1] * kernel[0][0] +	\
									img[i][j-1] * kernel[1][0] +	\
									img[i+1][j-1] * kernel[2][0] + 	\
									img[i-1][j] * kernel[0][1] + 	\
									img[i][j] * kernel[1][1] + 		\
									img[i+1][j] * kernel[2][1] + 	\
									img[i-1][j+1] * kernel[0][2] + 	\
									img[i][j+1] * kernel[1][2] + 	\
									img[i+1][j+1] * kernel[2][2]

	return output_image



def edge_detection_x(img):

	x_kernel = np.array([				
							[-1, 0, 1], 
							[-2, 0, 2], 
							[-1, 0 , 1] 
						], np.int8)  

	edge_x_img = convolve_img(img,x_kernel)
	print_image(edge_x_img,'edge_x_img')



def edge_detection_y(img):

	y_kernel = np.array([				
							[-1, -2, -1], 
							[0, 0, 0], 
							[1, 2 , 1] 
						], np.int8)  


	edge_y_img = convolve_img(img,y_kernel)
	print_image(edge_y_img,'edge_y_img')


def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )


def get_gaussian_kernel(sigma):

	kernel_radius = 1 

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

		gussian_blurred_img = convolve_img(img, get_gaussian_kernel(sigma_row[i]))
		write_image(gussian_blurred_img,'gb_img_'+ octav_id +'_'+str(i))




def generate_octavs(image_1,sigma_table):
	
	# octav 1: original image 

	generate_gaussian_blur_for_an_image(image_1,'octav_1', sigma_table[0])



	# octav 2: original image/2
	image_2 = resize_image_to_half(image_1)
	generate_gaussian_blur_for_an_image(image_2,'octav_2', sigma_table[1])




	# octav 3: original image/4
	image_3 = resize_image_to_half(image_2)
	generate_gaussian_blur_for_an_image(image_3,'octav_3', sigma_table[2])


	# octav 4: original image/8
	image_4 = resize_image_to_half(image_3)
	generate_gaussian_blur_for_an_image(image_4,'octav_4', sigma_table[3])


def compute_DoG():



	for i in range(0,4):
		for j in range(1,5):

			img_lower_blur = cv2.imread("gb_img_octav_" + str(j) + "_" + str(i) + ".png", 0)
			img_higher_blur = cv2.imread("gb_img_octav_" + str(j) + "_" + str(i+1) + ".png", 0)
			
			write_image(img_higher_blur-img_lower_blur,'Dog_octav'+ str(j)+'_'+ str(i))






def main():

	#task 1

	task_1_img = cv2.imread("task1.png", 0)
	# edge_detection_x(task_1_img)
	# edge_detection_y(task_1_img)


	#task 2

	task_2_img = cv2.imread("task2.jpg", 0)

	sigma_table = np.array([				
							[1/sqrt(2), 1, sqrt(2), 2, 2*sqrt(2)], 
							[sqrt(2), 2,  2*sqrt(2), 4, 4*sqrt(2)], 
							[2*sqrt(2), 4, 4*sqrt(2), 8, 8*sqrt(2)],
							[4*sqrt(2), 8, 8*sqrt(2), 16, 16*sqrt(2)]
						])

	

	#generate_octavs(task_2_img, sigma_table);
	compute_DoG()


	print('done!!!')





main()
 















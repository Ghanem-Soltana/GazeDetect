import cv2
import numpy as np
import json
from glob import glob
import math
import os
import shutil
import sys
images = "images"

json_fns = glob("images/*.json")

test = []
milleX = []
milleY = []
for json_fn in json_fns:

	img = cv2.imread("%s.jpg"%json_fn[:-5])
	data_file = open(json_fn)
	data = json.load(data_file)


	def milieu(x1,y1,x2,y2):
		x = (x1 + x2) / 2
		y = (y1+y2) /2
		return [x,y]
	#https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
	def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
		px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
					(x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
		py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
					(x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
		return [px, py]
	def process_json_list(json_list):
		ldmks = [eval(s) for s in json_list]
		return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

	ldmks_interior_margin = process_json_list( data['interior_margin_2d'])
	ldmks_caruncle = process_json_list( data['caruncle_2d'])
	ldmks_iris = process_json_list( data['iris_2d'])

	intersection = findIntersection(int(ldmks_interior_margin[0][0]),int(ldmks_interior_margin[0][1]),int(ldmks_interior_margin[round(len(ldmks_interior_margin)/2)][0]),int(ldmks_interior_margin[round(len(ldmks_interior_margin)/2)][1]),int(ldmks_interior_margin[4][0]),int(ldmks_interior_margin[4][1]),int(ldmks_interior_margin[12][0]),int(ldmks_interior_margin[12][1]))

	milieu_x = milieu(int(ldmks_interior_margin[0][0]),int(ldmks_interior_margin[0][1]),int(ldmks_interior_margin[round(len(ldmks_interior_margin)/2)][0]),int(ldmks_interior_margin[round(len(ldmks_interior_margin)/2)][1]))
	milieu_y = milieu(int(ldmks_interior_margin[4][0]),int(ldmks_interior_margin[4][1]),int(ldmks_interior_margin[12][0]),int(ldmks_interior_margin[12][1]))

	# Draw green foreground points and lines
	for ldmk in np.vstack([ldmks_interior_margin[0],ldmks_interior_margin[4],ldmks_interior_margin[12],ldmks_interior_margin[round(len(ldmks_interior_margin)/2)]]):
		cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0,255,0),-1)

	cv2.circle(img, (int(intersection[0]), int(intersection[1])), 2, (0,255,0),-1)
	cv2.circle(img, (int(milieu_x[0]), int(milieu_x[1])), 2, (0,255,0),-1)
	cv2.circle(img, (int(milieu_y[0]), int(milieu_y[1])), 2, (0,255,0),-1)

	look_vec = list(eval(data['eye_details']['look_vec']))

	eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
	print("begin point")
	print(tuple(eye_c))
	look_vec[1] = -look_vec[1]
	print("end point")
	print(tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int)))
	# origin
	point_A = tuple(eye_c)
	# horizon
	point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
	# where the eye look
	point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))

	# horizon
	cv2.line(img, point_A, point_B, (0, 0, 0), 3)
	cv2.line(img, point_A, point_B, (0, 255, 255), 2)
	# where the eye look
	cv2.line(img, point_A, point_C, (0, 0, 0), 3)
	cv2.line(img, point_A, point_C, (0, 255, 255), 2)

	angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
																					  point_B[1] - point_A[1]);

	angle = (angle * 180) / math.pi
	print("the angle")
	while (angle < 0):
		angle = angle + 360

	print(angle)

	print("distance point A et intersection")

	dist = math.sqrt( (point_A[0] - intersection[0]) * (point_A[0] - intersection[0]) + (point_A[1] - intersection[1]) * (point_A[1] - intersection[1]) )
	test.append(dist)
	dist_x = math.sqrt( (point_A[0] - milieu_x[0]) * (point_A[0] - milieu_x[0]) + (point_A[1] - milieu_x[1]) * (point_A[1] - milieu_x[1]) )
	dist_y = math.sqrt( (point_A[0] - milieu_y[0]) * (point_A[0] - milieu_y[0]) + (point_A[1] - milieu_y[1]) * (point_A[1] - milieu_y[1]) )

	milleX.append(dist_x)
	milleY.append(dist_y)

	print("max" + " x: " + str(max(milleX)) + " y: " + str(max(milleY)))
	print("max intersection:"+str(max(test)))

	cv2.imshow("syntheseyes_img", img)
	cv2.waitKey(150)

	print("choose eye direction")
	print("1-- Bottom left")
	print("2-- Bottom center")
	print("3-- Bottom right")
	print("4-- Middle left" )
	print("5-- Middle center")
	print("6-- Middle right")
	print("7-- Top left")
	print("8-- Top center")
	print("9-- Top right")
	print("10 to quit")


	stop = False
	while(not stop):
		choice = input("please select an option\n")
		try:
			stop = int(choice) in range(1, 11)
		except Exception:
			continue
	imageName = str(json_fn[7:-5]) + ".jpg"
	print("image Name")
	print(imageName)
	if choice == "4":
		classe = "MiddleLeft"
	if choice == "7":
		classe = "TopLeft"
	if choice == "8":
		classe = "TopCenter"
	if choice == "9":
		classe = "TopRight"
	if choice == "6":
		classe = "MiddleRight"
	if choice == "3":
		classe = "BottomRight"
	if choice == "2":
		classe = "BottomCenter"
	if choice == "1":
		classe = "BottomLeft"
	if choice == "4":
		classe = "MiddleLeft"
	if (choice == "5"):
		classe = "MiddleCenter"
	if (choice == "10"):
		sys.exit()
	print(json_fn)
	shutil.copyfile(images + "/" + imageName, "outputs/" + classe + "/" + imageName)
	shutil.copyfile(json_fn, "outputs/" + classe + "/" + str(json_fn[7:]))
	os.remove(images + "/" + imageName)
	data_file.close()
	os.remove(json_fn)

	print("==>" + classe)

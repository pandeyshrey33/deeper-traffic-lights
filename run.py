import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
import rospy
from rospy.numpy_msg import numpy_msg
		# %matplotlib inline

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from glob import glob

import time

CKPT = 'frozen_out/resnet-udacity-real-large-17675/frozen_inference_graph.pb'
PATH_TO_LABELS = 'tf_records_data/label_map.pbtxt'

NUM_CLASSES = 14

detection_graph = tf.Graph()

with detection_graph.as_default():

	od_graph_def = tf.GraphDef()

	with tf.gfile.GFile(CKPT, 'rb') as fid:

		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# print(category_index)

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		  (im_height, im_width, 3)).astype(np.uint8)

# IMAGE_SIZE = (12, 8)
# global detection_graph
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

count_left = 0
count_right = 0

queue_left = dict()
queue_right = dict()

position_x = 0
position_y = 0
position_z = 0
last_box = []
rotation_matrix = np.zeros((3, 3))
min_thresh = 0

current_position_x = 0
current_position_y = 0
current_position_z = 0

current_x = 0
current_y = 0
current_z = 0
current_w = 0


def getRotationMatrix(x, y, z, w):
	R = np.zeros((3, 3))
	
	R[0][0] = 1 - 2*y*y - 2*z*z
	R[0][1] = 2*x*y + 2*w*z
	R[0][2] = 2*x*z - 2*w*y

	R[1][0] = 2*x*y - 2*w*z
	R[1][1] = 1 - 2*x*x - 2*z*z
	R[1][2] = 2*y*z + 2*w*x

	R[2][0] = 2*x*z + 2*w*y
	R[2][1] = 2*y*z - 2*w*x
	R[2][2] = 1 - 2*x*x - 2*y*y

	return R


def setData(data):
	current_position_x = data.pose.pose.position.x
	current_position_y = date.pose.pose.position.y
	current_position_z = data.pose.pose.position_z

	current_x = data.pose.pose.orientation.x
	current_y = data.pose.pose.orientation.y
	current_z = data.pose.pose.orientation.z
	current_w = data.pose.pose.orientation.w

def img_cb_left(data):
	global count_left
	# getCurrentPosition()
	# if(changeInPosition < 1): #1 meter
		# return

	# global queue_left

	img = np.fromstring(data.data, np.uint8)
	encoded_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	image_np_expanded = np.expand_dims(encoded_img, axis=0)

	(boxes, scores, classes, num) = sess.run(
	  [detection_boxes, detection_scores, detection_classes, num_detections],
	  feed_dict={image_tensor: image_np_expanded})

	boxes = np.squeeze(boxes)
	scores = np.squeeze(scores)
	classes = np.squeeze(classes).astype(np.int32)
	count_left += 1
	if(boxes.shape[0] > 0):
		if(scores[0] > min_thresh):
			dataleft = [boxes, scores, classes]

			R = getRotationMatrix(current_x, current_y, current_z, current_w)
			
			rotation = np.matmul(R, rotation_matrix.transpose())
			translation = [current_position_x - position_x, current_position_y - position_y, current_position_z - position_z]

			posx1 = dataleft[0][i][0]*height
			posy1 = dataleft[position][0][i][1]*width

			posx2 = last_box[position][0][i][0]*height
			posy2 = last_box[position][0][i][1]*width

			print posx1, posy1
			print posx2, posy2

			ptx1 = (posx1)/fx
			ptx2 = (posx2)/fx

			pty1 = (posy1)/fy
			pty2 = (posy2)/fy

	# TODO : FILL IN THE MATRIX ENTRIES PROPERLY

			A[0][0] = rotation[0][0] - ptx1*rotation[2][0]
			A[0][1] = rotation[0][1] - ptx1*rotation[2][1]
			A[0][2] = rotation[0][2] - ptx1*rotation[2][2]

			A[1][0] = rotation[1][0] - pty1*rotation[2][0]
			A[1][1] = rotation[1][1] - pty1*rotation[2][1]
			A[1][2] = rotation[1][2] - pty1*rotation[2][2]

			A[2][0] = rotation[0][0] - ptx2*rotation[2][0]
			A[2][1] = rotation[0][1] - ptx2*rotation[2][1]
			A[2][2] = rotation[0][2] - ptx2*rotation[2][2]

			A[3][0] = rotation[1][0] - pty2*rotation[2][0]
			A[3][1] = rotation[1][1] - pty2*rotation[2][1]
			A[3][2] = rotation[1][2] - pty2*rotation[2][2]

			B[0] = translation[2]*ptx1 - translation[0]
			B[1] = translation[2]*pty1 - translation[1]
			B[2] = translation[2]*ptx2 - translation[0]
			B[3] = translation[2]*pty2 - translation[1]

			cv2.solve(A, B, X, cv2.DECOMP_SVD)
		  # X = np.linalg.lstsq(A, B, rcond=-1)[0]
			print A, B, X
			print "Estimated Distance", (X[0]**2 + X[1]**2 + X[2]**2)**0.5

			position_x = posx
			position_y = posy
			position_z = posz

			rotation_matrix = R
			last_box = dataleft

	# 				queue_left[count_left] = dataleft
	# 				if(count_left in queue_right):
	# 						cv2.imwrite("hello"+str(count_left)+"L.jpg", img)
	# 						getDistance(count_left)

# def getDistance(position):
# 	# print "HERE"
# 	height = 720
# 	width = 1280

# 	isLeft = False
# 	isRight = False

# 	fx = 0.679
# 	fy = 0.679

# 	num_eqs = 4
# 	num_vars = 3

# 	A = np.zeros((num_eqs, 3))
# 	B = np.zeros(num_eqs)
# 	X = np.zeros((3, 1))

# 	rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
# 	translation = [0 , 0 , 0.12]

# 	for i in xrange(1):
# 		posx1 = queue_left[position][0][i][0]*height
# 		posy1 = queue_left[position][0][i][1]*width

# 		posx2 = queue_right[position][0][i][0]*height
# 		posy2 = queue_right[position][0][i][1]*width

# 		print posx1, posy1
# 		print posx2, posy2

# 		ptx1 = (posx1)/fx
# 		ptx2 = (posx2)/fx

# 		pty1 = (posy1)/fy
# 		pty2 = (posy2)/fy

# 		A[0][0] = 1
# 		A[0][1] = 0
# 		A[0][2] = -ptx1

# 		A[1][0] = 0
# 		A[1][1] = 1
# 		A[1][2] = -pty1

# 		A[2][0] = 1
# 		A[2][1] = 0
# 		A[2][2] = -ptx2

# 		A[3][0] = 0
# 		A[3][1] = 1
# 		A[3][2] = rotation[5] - pty2*rotation[8];

# 		B[0] = translation[2]*ptx1 - translation[0]
# 		B[1] = translation[2]*pty1 - translation[1]
# 		B[2] = translation[2]*ptx2 - translation[0]
# 		B[3] = translation[2]*pty2 - translation[1]

# 		cv2.solve(A, B, X, cv2.DECOMP_SVD)
# 	  # X = np.linalg.lstsq(A, B, rcond=-1)[0]
# 		print A, B, X
# 		print "Estimated Distance", (X[0]**2 + X[1]**2 + X[2]**2)**0.5

rotation_matrix = getRotationMatrix(0, 0, 0, 1)

rospy.init_node("Hi")
# img_sub_right = rospy.Subscriber('/zed/right/image_rect_color/compressed', numpy_msg(CompressedImage), img_cb_right, queue_size=1)
img_sub_left = rospy.Subscriber('/zed/left/image_rect_color/compressed', numpy_msg(CompressedImage), img_cb_left, queue_size=1)
odom_sub = rospy.Subscriber('/zed/odom', Odometry, setData)
rospy.spin()

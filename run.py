import numpy as np
import os
import sys
import tensorflow as tf
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from sensor_msgs.msg import CompressedImage, Image
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

# def img_cb(data):
# PATH_TO_TEST_IMAGES_DIR = 'test_images_udacity'

# print(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
# TEST_IMAGE_PATHS = glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
# print("Length of test images:", len(TEST_IMAGE_PATHS))

IMAGE_SIZE = (12, 8)
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

count = 0

def img_cb(data):
	# for image_path in TEST_IMAGE_PATHS:
	global count
	# print "Hi"
	time0 = time.time()
	img = np.fromstring(data.data, np.uint8)
	encoded_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	# enocded_img = cv2.resize(encoded_img, (0, 0), fx = 0.5, fy = 0.5)
	# print encoded_img.shape
	# np.reshape(encoded_img, (720, 1280, 3))
	# print img.shape
	# image = Image.open(image_path)
	# # the array based representation of the image will be used later in order to prepare the
	# # result image with boxes and labels on it.
	# image_np = load_image_into_numpy_array(image)
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(encoded_img, axis=0)
	# Actual detection.
	(boxes, scores, classes, num) = sess.run(
	  [detection_boxes, detection_scores, detection_classes, num_detections],
	  feed_dict={image_tensor: image_np_expanded})

	boxes = np.squeeze(boxes)
	scores = np.squeeze(scores)
	classes = np.squeeze(classes).astype(np.int32)

	# Visualization of the results of a detection.
	# vis_util.visualize_boxes_and_labels_on_image_array(
	#     image_np, boxes, classes, scores,
	#     category_index,
	#     use_normalized_coordinates=True,
	#     line_thickness=6)

	# plt.figure(figsize=IMAGE_SIZE)
	# plt.imshow(encoded_img)
	# plt.show()

	min_score_thresh = .50
	for i in xrange(boxes.shape[0]):
		if scores is None or scores[i] > min_score_thresh:

			class_name = category_index[classes[i]]['name']
			print('{}'.format(class_name), scores[i])
			
			fx =  1345.200806
			fy =  1353.838257
			
			perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
			perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600

			# ymin, xmin, ymax, xmax = box
			# depth_prime = (width_real * focal) / perceived_width
			perceived_depth_x = ((.1 * fx) / perceived_width_x)
			perceived_depth_y = ((.3 * fy) / perceived_width_y)

			estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)
			print("Distance (metres)", estimated_distance)
			cv2.imwrite('img'+str(count)+str(class_name)+str(estimated_distance)+'.jpg', encoded_img)
	time1 = time.time()
	print("Time in milliseconds", (time1 - time0) * 1000) 
	# count += 1


rospy.init_node("Hi")
img_sub = rospy.Subscriber('/zed/right/image_rect_color/compressed', numpy_msg(CompressedImage), img_cb, queue_size=1)
rospy.spin()



Install instructions:

1) Tensorflow object detection API
2) Download pre-trained network
3) Tensorflow Models directory
4) Create Directory 'frozen_out'. Download resnet-udacity-real-large-17675. Look it up.
5) From models (tensorflow) copy the object_detection folder into the main directory.


run.py uses data from two cameras and uses this to extrapolate the location of the given bounding box in coordinates wrt the camera. (Subscribes to /zed/right/image_rect_color/compressed, /zed/left/image_rect_color/compressed.)

newrun.py uses data from one camera at different instances of time to do the same. (Extra dependancy on Odometry data)

Clone, run 'python run.py'. Outputs images in the home directory with detected colour and distance in the name of the file.

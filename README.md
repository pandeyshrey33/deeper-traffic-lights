This package detects traffic lights and the distance of the traffic lights using two different aproaches, both using the deeper-traffic-lights model to get a tight bounding box on the traffic light.

Two Camera Approach :

This approach uses the fact that multiplying the camera coordinates with the inverse of the Camera Matrix gives us the ray along which the point lies. Using two rays, we can find the point of intersection of rays. This intersection needs the location of one of the cameras WRT the other. Since we have 4 equations and 3 variables, we use a linear least square approach to get the best estimate for the 3D coordinates wrt to the camera.

This approach requires the two cameras to be located such that the system of equations does not become highly sensitive.

Single Camera Approach :

This approach is essentially same as the previous approach. This uses only one camera but the location of the traffic light at two instances of time is used to compute the 3D coordinates of the traffic light WRT the camera.

This requires good Odometry data.


Install instructions:

1) Tensorflow object detection API
2) Download pre-trained network
3) Tensorflow Models directory
4) Create Directory 'frozen_out'. Download resnet-udacity-real-large-17675. Look it up.
5) From models (tensorflow) copy the object_detection folder into the main directory.


run.py uses data from two cameras and uses this to extrapolate the location of the given bounding box in coordinates wrt the camera. (Subscribes to /zed/right/image_rect_color/compressed, /zed/left/image_rect_color/compressed.)

newrun.py uses data from one camera at different instances of time to do the same. (Extra dependency on Odometry data)

Clone, run 'python run.py'. Outputs images in the home directory with detected colour and distance in the name of the file.

#############################################
# 
# Author: Tien Ha Chu
# Date: 2018-09-01
# BinaryImageRegistration requirements
#
#############################################

Requirement: OpenCV2 or OpenCV3

The following parts are used to register binary images:

ExtractContours
Extracts contour from canny image.

BinaryRegistration
Performs binary image registration algorithm in "Binary image registration using covariant gaussian densities. Csaba Domokos and Zoltan Kato. In Image analysis and recognition: 5th international conference, ICIAR 2008, 2008, only being tested for rotation case.
The executable rotates template image around its centre a specific angle to get observed image, runs registration algorithm to find homography and warpes the template
image with the computed homography, the warped image is finally shown.
Usage: ./BinReg alpha src_img dst_img
alpha: rotation angle
src_img: template image
dst_img: observed image
example:
./BinReg 0.1576 ../data/whole_gear_thresholded.ppm ../data/whole_gear_canny_edged.ppm

Test script:
The test script creates tested images by continuously rotating the object around its centre pi/10, 2*pi/10... 20*pi/10, finds the homography which maps template image to observed image, apply the homography transformation on the template image, measures registration errors and writes output to file.
Place this script in the same folder as the executable ./BinaryImageRegistration and run it with the following syntax:
./iterative_registration_test.sh thresholded_img canny_img
example: ./iterative_registration_test.sh ../data/whole_gear_thresholded.ppm ../data/whole_gear_canny_edged.ppm


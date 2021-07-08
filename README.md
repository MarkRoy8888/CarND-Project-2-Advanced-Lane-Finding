# Project2
My Advanced-Lane-Lines Project2

In this project, My goal is to find the Lane and apply to video.

---
### The goals / steps of this project are the following:
#### 1.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. <br />
#### 2.Apply a distortion correction to raw images.<br />
#### 3.Use color transforms, gradients, etc., to create a thresholded binary image.<br />
#### 4.Apply a perspective transform to rectify binary image ("birds-eye view").<br />
#### 5.Detect lane pixels and fit to find the lane boundary.<br />
#### 6.Determine the curvature of the lane and vehicle position with respect to center.<br />
#### 7.Warp the detected lane boundaries back onto the original image.<br />
#### 8.Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.<br />
---

### 1.Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. <br />

#### chose the image can find corner
![alt-text-1](readme/output_16_1.png "Corners detected")
#### there are some image don't find corner
![alt-text-1](readme/output_16_3.png "Unable to detect corners")

### 2.Apply a distortion correction to raw images.<br />

![alt-text-1](readme/output_31_0.png "Undistortion | distortion")


### 3.Use color transforms, gradients, etc., to create a thresholded binary image.<br />
#### use color gradients to find the line
![alt-text-1](readme/output_29_0.png "original | color transforms")
![alt-text-1](readme/output_35_0.png "original | color transforms add sobel")

### 4.Apply a perspective transform to rectify binary image ("birds-eye view").<br />
![alt-text-1](readme/output_33_0.png "original | perspective")



### 5.Detect lane pixels and fit to find the lane boundary.<br />
![alt-text-1](readme/output_42_3.png "Detect lane pixels")


### 6.Determine the curvature of the lane and vehicle position with respect to center.<br />
![alt-text-1](readme/output_45_1.png "add curvature")
### 7.Warp the detected lane boundaries back onto the original image.<br />

### 8.Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.<br />
![alt-text-1](readme/output_45_1.png "Warp")
![alt-text-1](readme/output_47_0.png "Warp")
![alt-text-1](readme/output_47_1.png "Warp")
![alt-text-1](readme/output_47_2.png "Warp")
![alt-text-1](readme/output_47_3.png "Warp")
![alt-text-1](readme/output_47_4.png "Warp")
![alt-text-1](readme/output_47_5.png "Warp")

#### And apply to video, more in the file
showing challenge video, it seems ok.

![image](https://github.com/MarkRoy8888/CarND-Project-2-Advanced-Lane-Finding/blob/master/output_video/project2-challeng2.gif)


### Discussion
#### 1.Seting different situation Model maybe having better Result.
#### 2.Different  situation having different color situation
#### 3.Avoiding change too much, follow the last data to edit.

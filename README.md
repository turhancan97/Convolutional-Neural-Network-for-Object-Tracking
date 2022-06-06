
# Convolutional Neural Network for Object Tracking
* **Author:** [Turhan Can Kargın](https://github.com/turhancan97) and [Kamil Gültekin](https://github.com/kamilgultekin)
* **Topic:** Vision-Based Control Course Project

The project we prepared for the **Vision-based Control** lecture, which is one of the Poznan University of Technology Automatic Control and Robotics graduate courses. In this project, you will find couple of projects that can be helpful to learn Image Processing, Computer Vision, Deep Neural Network, Convolutional Neural Network topics and OpenCV, Tensorflow, Keras, YOLO Frameworks.

# Table of Contents
   * [Summary](#summary)
   * [Introduction](#introduction)
	   * [Image Processing](#image-processing)
		   * [Image Processing Techniques](#image-processing-techniques)
	   * [Computer Vision](#computer-vision)
		   * [Conventional Computer Vision](#conventional-computer-vision)
		   * [Deep Learning based Computer Vision](#deep-learning-based-computer-vision)
	   * [Difference between Image Processing  and Computer Vision](#difference-between-image-processing-and-computer-vision)
	   * [Deep Neural Networks](#deep-neural-networks)
		   * [Pytorch](#pytorch)
		   * [Tensorflow and Keras](#tensorflow-and-keras)
		   * [FastAi](#fastai)
	   * [Convolutional Neural Networks](#convolutional-neural-networks)
		   * [R-CNN](#r-cnn)
		   * [Fast R-CNN](#fast-r-cnn)
		   * [Faster R-CNN](#faster-r-cnn)
		   * [Mask R-CNN](#faster-r-cnn)
		   * [YOLO](#r--cnn)
* [Object Detection](#object-detection)
	* [Examples](#example-detection)
* [Object Recognition](#object-recognition)
	* [Examples](#example-recognition)
* [Object Tracking](#object-tracking)
	* [Examples](#example-tracking)
* [Project Lists](#project-lists)
	* [Object Detection with Specific Color](#object-detection-with-specific-color)
	* [Motion Detection and Tracking with OpenCV](#motion-detection-and-tracking-with-opencv)
	* [Face and Eye Detection with OpenCV and Haar Feature based Cascade Classifiers](#face-and-eye-detection)
	* [MNIST Image Classification with Deep Neural Network](#mnist-image-classification-with-deep-neural-network)
	* [MNIST Image Classification with Convolutional Neural Network](#mnist-image-classification-with-convolutional-neural-network)
	* [Face Mask Detection for COVID19 with OpenCV, Tensorflow 2 and Keras](#face-mask-detection-for-covid19)
	* [Object Detection and Tracking in Custom Datasets with Yolov4](#object-detection-and-tracking-in-custom-datasets-with-yolov4)
* [Future Work](#future-work)
* [References](#references)

# Summary
pass
# Introduction

pass

## Image Processing
**Image processing** is a method that can be identified with different techniques in order to obtain useful information according to the relevant need through the images transferred to the digital media. The image processing method is used to process certain recorded images and modify existing images and graphics to alienate or improve them. For example, it is possible to see the quality declines when scanning photos or documents and transferring them to digital media. This is where the image processing method comes into play during the quality declines. We use image processing method to minimize the degraded image quality and visual distortions. Another example is the maps that we can access via Google Earth. Satellite images are enhanced by image processing techniques. In this way, people are presented with higher quality images. Image processing, which can be used in this and many other places, is among the rapidly developing technologies. It is also one of the main research areas of disciplines such as engineering and computer science [1].
![Figure 1 A normal landscape photo and its gray scale](https://user-images.githubusercontent.com/22428774/171999676-3c4dad8b-786d-4084-84b3-496e58b40e0a.png)
Image processing is basically studied in three steps.

* Transferring the image with the necessary tools
* Analyzing the image and processing it in the desired direction
* Receiving the results of the data report and output that are analyzed and processed

In addition to the image processing steps, two types of methods are used for image processing. The first is analog image processing and the other is digital image processing. There are a number of basic steps that data must go through for digital and analog image processing. These steps are as follows:
* Preprocessing
* Development and viewing
* Information extraction

After these steps, results can be obtained from the relevant data according to the needs.

Image processing has purposes such as visualization, making partially visible objects visible, image enhancement, removing spots, noise removal, high resolution capture in the image, pattern, and shape recognition.


### Image Processing Techniques
Some techniques which are used in digital image processing include: 
* Anisotropic diffusion 
* Hidden Markov models 
* Image editing 
* Image restoration 
* Independent component analysis 
* Linear filtering 
* Neural networks 
* Partial differential equations 
* Pixelation 
* Point feature matching 
* Principal components analysis 
* Self-organizing maps 
* Wavelets

## Computer Vision

Human perception of the outside world is formed by perceiving and analyzing images, which is one of the most important perception channels, and interpreting them. All techniques aiming to create visual perception and understanding in humans on the computer fall into the field of computer vision and cognitive learning. Scientists working in the field of computer vision have played the biggest role in the advancement of artificial intelligence, which is the most mentioned deep learning sub-field today. 

Computer vision aims to make sense of all kinds of two-dimensional, three-dimensional or higher-dimensional visual digital data, especially with smart algorithms. In solving these problems, the field of computer vision uses the development and implementation of both mathematical and computational theory-based techniques such as geometry, linear algebra, probability and statistics theory, differential equations, graph theory, and in recent years, especially machine learning and deep learning. In addition to standard camera images, medical images, satellite images, and computer modeling of three-dimensional objects and scenes are of interest to computer vision.

Visual data detection and interpretation constitute the most important algorithm steps in many applications, from autonomous systems such as autonomous vehicles, robots, drones, to security and biometric verification areas. The outputs obtained are fed as inputs to the decision support systems in the next step, completing the artificial intelligence systems [2].
![Vision Based Control (1)](https://user-images.githubusercontent.com/22428774/172026103-78ca3882-ff50-4e7a-aa47-27543a0a1e5b.png)

### Conventional Computer Vision

pass

### Deep Learning based Computer Vision

pass

## Difference between Image Processing  and Computer Vision
pass
## Deep Neural Networks
pass
###  Pytorch
pass
### Tensorflow and Keras
pass
### FastAi
pass
## Convolutional Neural Networks
pass
###  R-CNN
pass
### Fast R-CNN
pass
### Faster R-CNN
pass
### Mask R-CNN
pass
### Yolo
pass

# Object Detection

pass

## Example Detection
pass
# Object Recognition

pass


## Example Recognition

# Object Tracking

pass

## Example Tracking

pass

# Project List

1. Object Detection with Specific Color
2. Motion Detection and Tracking with OpenCV
3. Face and Eye Detection with OpenCV and Haar Feature based Cascade Classifiers
4. MNIST Image Classification with Deep Neural Network
5. MNIST Image Classification with Convolutional Neural Network
6. Face Mask Detection for COVID19 with OpenCV, Tensorflow 2 and Keras
7. Object Detection and Tracking in Custom Datasets with Yolov4

## Object Detection with Specific Color
The goal here is fair self-explanatory:
-   **Step #1:** Detect the presence of a colored object (blue in the code) using computer vision techniques.
-   **Step #2:** Track the object as it moves around in the video frames, drawing its previous positions as it moves.

The end product should look similar to the GIF below:

![color_detection](https://user-images.githubusercontent.com/22428774/172240232-ab67f8d5-3fc1-4255-b5ab-e918e82cd33f.gif)


Using HSV color range which is determined as Lower and Upper, I detected colorful object. Here I prefered blue objects.

```python
# blue HSV
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)
```
When I got the color range, I set capture size and then I read the capture.

First I apply Gaussian Blurring for decreasing the noises and details in capture. 
```python
#blur
blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0)
```
After Gaussian Blurring, I convert that into HSV color format.
```python
# HSV
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
```
To detect Blue Object, I define a mask.
```python
# mask for blue
mask = cv2.inRange(hsv, blueLower, blueUpper)
```
After mask, I have to clean around of masked object. Therefor I apply first Erosion and then Dilation
```python
# deleting noises which are in area of mask
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
```
After removing noises, the Contours have to be found
```python
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center = None
```
If the Contours have been found, I'll get the biggest contour due to be well.
```python
# get max contour
c = max(contours, key=cv2.contourArea)
```
The Contours which are found have to be turned into rectangle deu to put rectangle their around. This cv2.minAreaRect() function returns a rectangle which is smallest to cover the area of object.
```python
rect = cv2.minAreaRect(c)
```
In the screen, we want to print the information of rectangle, therefore we need to reach its inform.
```python
((x,y), (width, height), rotation) = rect
s = f"x {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
```
Using this rectangle we found, we want to get a Box. In the next, we will use this Box for drawing Rectangle.
```python
# box
box = cv2.boxPoints(rect)
box = np.int64(box)
```
Image Moment is a certain particular weighted average (moment) of the image pixels' intensities. To find Momentum, we use Max. Contour named as "c". After that, I find Center point.
```python
# moment
M = cv2.moments(c)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
```
Now, I will draw the center which is found.
```python
# point in center
cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
```
After Center Point, I draw Contour
```python
# draw contour
cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
```
We want to print coordinators etc. in the screen 
```python
# print inform
cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
```

Finally, we wrote the code below to track the blue object with past data
```python
# deque - draw the past data
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # 
        cv2.imshow("Original Detection",imgOriginal)
```
## Motion Detection and Tracking with OpenCV
pass
## Face and Eye Detection
pass
## MNIST Image Classification with Deep Neural Network
pass
## MNIST Image Classification with Convolutional Neural Network
pass
## Face Mask Detection for COVID19
pass
## Object Detection and Tracking in Custom Datasets with Yolov4
pass

# Future Work
pass
# References
- [1] - https://peakup.org/blog/yeni-baslayanlar-icin-goruntu-islemeye-giris/
- [2] - https://yapayzeka.itu.edu.tr/arastirma/bilgisayarla-goru
- [3] - VBM686 – Computer Vision Pinar Duygulu Hacettepe University Lecture Notes


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
		   * [Traditional Computer Vision](#traditional-computer-vision)
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

### Traditional Computer Vision
Computer vision has emerged as a critical research topic, with commercial applications based on computer vision approaches accounting for a significant share of the market. Over the years, the accuracy and speed with which images acquired by cameras are processed and identified has improved. Deep learning, being the most well-known lad in town, is playing a vital role as a computer vision technology.

**Is deep learning the only way to accomplish computer vision?**
No, no, no! A few years ago, deep learning made its debut in the field of computer vision. Image processing algorithms and approaches were the mainstays of computer vision at the time. The main task of computer vision was to extract the image's features. When doing a computer vision task, the initial stage was to detect color, edges, corners, and objects. These features are human-engineered, and the extracted features, as well as the methodologies employed for feature extraction, have a direct impact on the model's accuracy and reliability. In the traditional vision scope, the algorithms like SIFT _(Scale-Invariant Feature Transform)_, SURF _(Speeded-Up Robust Features)_, BRIEF _(Binary Robust Independent Elementary Features)_ plays the major role of extracting the features from the raw image.

The difficulty with this approach of feature extraction in image classification is that you have to choose which features to look for in each given image. When the number of classes of the classification goes high or the image clarity goes down it’s really hard to cope up with traditional computer vision algorithms.

### Deep Learning based Computer Vision

In the field of computer vision, deep learning, which is a subset of machine learning, has shown considerable performance and accuracy gains. In a popular ImageNet computer vision competition in 2012, a neural network with over 60 million parameters greatly outperformed previous state-of-the-art algorithms to picture recognition, arguably one of the most influential studies in bringing deep learning to computer vision.

![Vision Based Control](https://user-images.githubusercontent.com/22428774/172243755-faa685ad-dd76-4f2b-9cf3-e015bb1faf62.png)


The boom started with the convolutional neural networks and the modified architectures of ConvNets. By now it is said that some convNet architectures are so close to 100% accuracy of image classification challenges, sometimes beating the human eye!

The main difference in deep learning approach of computer vision is the concept of end-to-end learning. There’s no longer need of defining the features and do feature engineering. The neural do that for you. It can simply put in this way. Though deep neural networks has its major drawbacks like, need of having huge amount of training data and need of large computation power, the field of computer vision has already conquered by this amazing tool already!


## Difference between Image Processing  and Computer Vision
Computer vision and image processing are both attractive fields of computer science.

In **computer vision**, computers or machines are programmed to extract high-level information from digital images or videos as input, with the goal of automating operations that the human visual system can perform. It employs a variety of techniques, including Image Processing.

**Image processing** is the science of enhancing photographs by adjusting a variety of parameters and attributes. As a result, Image Processing is considered a subset of Computer Vision. In this case, transformations are made to an input image, and the resulting image is returned. Sharpening, smoothing, stretching, and other changes are examples of these transformations.

Both of the fields is working with visuals, i.e., images and videos. In fact, although it is not very accurate, we can say that if you use artificial intelligence algorithms and image processing methods in a project, that project is probably turning into a Computer Vision project. So Computer Vision is an intersection of Artificial Intelligence and Image Processing that usually aims to simulate intelligent human abilities.

![Vision Based Control (1)](https://user-images.githubusercontent.com/22428774/172245057-e60fd3ef-86ca-4af5-b209-4012fa6363ee.png)

|Image Processing|Computer Vision|
|-----|--------|
|Image processing is mainly focused on processing the raw input images to enhance them or preparing them to do other tasks|Computer vision is focused on extracting information from the input images or videos to have a proper understanding of them to predict the visual input like human brain.       |
|Image processing uses methods like Anisotropic diffusion, Hidden Markov models, Independent component analysis, Different Filtering etc.|Image processing is one of the methods that is used for computer vision along with other Machine learning techniques, CNN etc.      |
|Image Processing is a subset of Computer Vision.|Computer Vision is a superset of Image Processing.       |
|Examples of some Image Processing applications are- Rescaling image (Digital Zoom), Correcting illumination, Changing tones etc.|Examples of some Computer Vision applications are- Object detection, Face detection, Hand writing recognition etc.       |

## Deep Neural Networks

Deep neural networks are a field of research in which researchers show great interest under the science of artificial intelligence. It covers studies on learning computers. In this section, firstly, a general introduction to artificial intelligence will be given, then deep neural networks will be examined, and then the most widely used deep learning frameworks will be examined.

Computers and computer systems have become an indispensable part of life in the contemporary world. Many devices, from mobile phones to refrigerators in kitchens, work with computer systems. It has become commonplace to use computers in almost every field, from the business world to public affairs, from environmental and health organizations to military systems. When the development of technology is followed, it is seen that computers, which were previously developed only for electronic data transfer and performing complex calculations, gain qualifications that can filter and summarize large amounts of data over time and make comments about events using existing information. Today, computers can both make decisions about events and learn the relationships between events. 

Problems that cannot be formulated mathematically and cannot be solved can be solved by computers using heuristic methods. Studies that equip computers with these features and enable them to develop these abilities are known as "artificial intelligence" studies. Artificial intelligence, in its simplest definition, is the general name of systems that try to imitate a certain part of human intelligence. From this point of view, when it comes to artificial intelligence, we should not think of systems that can completely imitate human intelligence or that have this purpose.

Artificial intelligence can show itself in many different areas: Systems that predict what we write in our daily life, the search engine that allows us to search for an image on Google, Youtube's video recommendation system, and Instagram, which is very curious about how it once worked, is frequently ranked by those who see the story. are the examples we encountered.

![Vision Based Control](https://user-images.githubusercontent.com/22428774/173669486-c2a73b96-b520-4099-93d8-b3c95dd8aec8.png)

Deep learning, a sub-branch of Artificial Intelligence, is simply the name we give to training multi-layer artificial neural networks (Multi Layer Artificial Neural Networks) with an algorithm called "backpropagation". Even these two concepts are broad concepts that can be explained by books on their own. Artificial neural networks (ANNs), which are used in deep learning, are computer software that perform basic functions such as learning, remembering, and generating new data from the data it collects by imitating the learning path of the human brain. Artificial neural networks, inspired by the human brain, emerged as a result of the mathematical modeling of the learning process.

Biological neuron and artificial neural network simulations are given in Figure 5. Biological Nervous system elements and their equivalents in the artificial nervous system are given in table below. Here, the biological nervous system is divided into parts and each element is given an equivalent in the artificial neural network system.

|Biological Neuron System|Artificial NeuronSystem|
|-----|--------|
|Neuron|Processor Element|
|Dendrite| Aggregation Function|
|Cell Body|Transfer Function|
|Axons|Artificial Neuron Output|
|Synapses|Weights|

As seen in below gift, dog and cat data enters the network. The data processed in the middle layers is sent from there to the output layer. In short, this process is the conversion of incoming information to the output using the weight values of the network. In order for the network to produce the correct outputs for the inputs, the weights must have the correct values. 
![212a609564e5e4f0403ab4c671a4f80d](https://user-images.githubusercontent.com/22428774/173663632-cfac59b0-2bdf-41c7-9fca-93a5471f98c3.gif)

If you want to get more detailed information about artificial neural networks, [you can click on this link](https://github.com/turhancan97/Neural-Network-Training-on-Matlab). Let's take a look at the most widely used deep learning frameworks.
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
- [4] - O’Mahony, N.; Campbell, S.; Carvalho, A.; Harapanahalli, S.; Velasco-Hernandez, G.; Krpalkova, L.; Riordan, D.; Walsh, J. Deep Learning vs. Traditional Computer Vision. arXiv 2019, arXiv:1910.13796.
- [5] - https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7

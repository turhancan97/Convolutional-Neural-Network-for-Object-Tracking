# Object Detection with Color
import cv2
import numpy as np
from collections import deque

# data type to store object center
buffer_size = 16
pts = deque(maxlen = buffer_size)


# blue color range HSV
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

# capture
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success, imgOriginal = cap.read()
    
    if success: 
        
        # blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",hsv)
        
        # create mask for blue
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image",mask)
        # delete the noises around the mask
        mask = cv2.erode(mask, None, iterations = 3)
        mask = cv2.dilate(mask, None, iterations = 3)
        cv2.imshow("Mask + erozyon ve genisleme",mask)
        
        # for different version
        # (_, contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # contours
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            
            # take the biggest contour
            c = max(contours, key = cv2.contourArea)
            
            # convert to rectangle
            rect = cv2.minAreaRect(c)
            
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            # Box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # draw contour: yellow
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)
            
            # let's draw a dot on the center: pink
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
            
            # print information to screen
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        # deque - draw the past data
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # 
            
        cv2.imshow("Original Detection",imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break
# Very Basic Motion Detection and Tracking Using Opencv Contours. 
# We will see what contours are. we will Learn to find contours, draw contours, we will see these functions : cv2.findContours(), cv2.drawContours(). 
# In this project we are detecting and tracking motion using live sample video.
# The function retrieves contours from the binary image. The contours are a useful tool for shape analysis and object detection and recognition. 

import cv2

print("Choose a City among Poznan, Paris or Rome")
print("-------------------------------------------")

cap = cv2.VideoCapture('poznan.mp4') # Open the Video


frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read() # We define two frame one after another
ret, frame2 = cap.read()

print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) # To find out absolute difference of first frame and second frame
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # convert it to gray scale - We do it for contour stages (It is easier to find contour with gray scale)
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Blur the grayscale frame
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # max threshold values is 255 - we need trashold value
    dilated = cv2.dilate(thresh, None, iterations=5) 
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contour

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3) # If something is moving in the video then we will see Status: Movement
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
out.release()
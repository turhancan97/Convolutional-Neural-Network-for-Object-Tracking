## How we can do Face Detection using Haar Feature based Cascade Classifiers.
## It is a ML based approach and cascade function is trained for a lot of images

# A classifier (namely a cascade of boosted classifiers working with haar-likefeatures) is trained a 
# few hundred sample views of a particular object (car, face),
# called positive examples, that are scaled to the same size (e.g. 20x20), and negartive examples - 
# arbitrary images of the same size

# Because we will detect face and eyes we should import eye and face images

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

print("Enter Female Face or Male Face")
cap = cv2.VideoCapture('female.mp4')

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey ,ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
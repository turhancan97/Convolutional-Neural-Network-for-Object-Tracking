import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pickle

path = "digits_data"

myList = os.listdir(path)
noOfClasses = len(myList)

print("Number of Label/Class: ",noOfClasses)


images = []
classNo = []

#### IMPORTING DATA/IMAGES FROM FOLDERS 
for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\"+str(i))
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)
        
print(len(images))
print(len(classNo))

#### CONVERT TO NUMPY ARRAY
images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

#### SPLITTING THE DATA
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.3, random_state = 42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)


#### PLOT BAR CHART FOR DISTRIBUTION OF IMAGES
# # vis 
# fig, axes = plt.subplots(3,1,figsize=(7,7))
# fig.subplots_adjust(hspace = 0.5)
# sns.countplot(y_train, ax = axes[0])
# axes[0].set_title("y_train")

# sns.countplot(y_test, ax = axes[1])
# axes[1].set_title("y_test")

# sns.countplot(y_validation, ax = axes[2])
# axes[2].set_title("y_validation")

#### PREPOSSESSING FUNCTION FOR IMAGES FOR TRAINING 
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255
    
    return img


# idx = 311
# img = preProcess(x_train[idx])
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocess ",img)
    
x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

#### RESHAPE IMAGES 
x_train = x_train.reshape(-1,32,32,1)
print(x_train.shape)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

#### IMAGE AUGMENTATION 
dataGen = ImageDataGenerator(width_shift_range = 0.08,
                             height_shift_range = 0.08,
                             zoom_range = 0.08,
                             rotation_range = 8)

dataGen.fit(x_train)

#### ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

#### CREATING THE MODEL 
model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D( filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation = "relu" ))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses, activation = "softmax" ))

model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

batch_size = 64

#### STARTING THE TRAINING PROCESS
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 30,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)


#### SAVE THE TRAINED MODEL 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_trained.h5")
print("Saved model to disk")

# %% #### PLOT THE RESULTS  
hist.history.keys()


plt.figure()
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Train accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation accuracy")
plt.legend()
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(x_test, y_test, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

#### Confusion Matrix
y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
Y_true = np.argmax(y_validation, axis = 1)
cm = confusion_matrix(Y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()
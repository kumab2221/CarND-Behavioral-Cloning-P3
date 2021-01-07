import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
images = []
measurements = []
augmented_images = []
augmented_measurements = []

car_images = []
steering_angles = []

csv_file = './CarND-Behavioral-Cloning-P3/data/driving_log.csv'
path = './CarND-Behavioral-Cloning-P3/data/IMG/'

with open(csv_file, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        filename = row[0].split('\\')[-1]
        current_path = path + filename
        img_center = cv2.imread(current_path)
        images.append(img_center)
        measurements.append(steering_center)

        filename = row[1].split('\\')[-1]
        current_path = path + filename
        img_left = cv2.imread(current_path)
        images.append(img_left)
        measurements.append(steering_left)

        filename = row[2].split('\\')[-1]
        current_path = path + filename
        img_right = cv2.imread(current_path)
        images.append(img_right)
        measurements.append(steering_right)

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x /255.0 -0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((65,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)
model.save('./CarND-Behavioral-Cloning-P3/model.h5')

### print the keys contained in the history object
print(history.history.keys())
print(history.history['loss'])
print(history.history['val_loss'])
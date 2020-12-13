# This code is executed on Google Colab
# The dataset for this program is downloaded from CIFAR10

# Loading the libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Loading the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Look at the datatypes of the variable
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# Look at the shape of arrays
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

# Look at the actual image
index = 9
img = plt.imshow(x_train[index])

# Look at the img as an array
x_train[index]

# Get the image label
print("The image labe is: ", y_train[index])

# Get the image classification
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Print the classification
print('The image classification is : ', classification[y_train[index][0]])

# Convert the label into the set of 10 numbers to input to the neural network.
# It sets the category of the image it belongs to as 1 and assigns rest as 0.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels
print(y_train_one_hot)
print(y_test_one_hot)

# Print new label for the current image at index = 9
print("The one hot label is: ", y_train_one_hot[index])

# Normalize the pixels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255
# Printing the normalized value
x_train[index]


# Create model's architecture
model = Sequential()

# Add the first convolutional layer
# conv2D is used to create a convolutional kernal from input to generate tensors of output
# Here we use relu activation function so that it does not activate all the neurons at the same time
model.add(Conv2D(32, (5,5), activation='relu', input_shape = (32,32,3)))

# Add a pooling layer
# Mac Pooling is used to reduce the dimensionality of the input
model.add(MaxPooling2D(pool_size = (2,2)))

# Add  another convolutional layer
model.add(Conv2D(32, (5,5), activation='relu'))

# Add a pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

# Add a flattening layer to covert the data into 1-dimensional array for input to the next layer
model.add(Flatten())

# Add a layer 1000 neurons
model.add(Dense(1000, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer 500 neurons
model.add(Dense(500, activation='relu'))

# Add a dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Add a layer 250 neurons
model.add(Dense(250, activation='relu'))

# Add a layer 10 neurons
# Softmax activation function  The input values can be positive, negative, zero, or greater than one,
# but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities.
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)

# Evaluate the model using test data
model.evaluate(x_test, y_test_one_hot)[1]

# Visualize the model accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train',' Val'], loc = 'upper left')
plt.show

# Visualize the model loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train',' Val'], loc = 'upper right')
plt.show

# Test the model

from google.colab import files
uploaded = files.upload()

# Show the image
new_img = plt.imread('images.jpg')
img = plt.imshow(new_img)

# Resize the uploaded image
from skimage.transform import resize
resized_img = resize(new_img, (32,32,3))
img = plt.imshow(resized_img)

# Get the model prediction
prediction = model.predict(np.array([resized_img]))
prediction

# Sort the predictions
list_index = [0,1,2,3,4,5,6,7,8,9]
x = prediction

for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

# Show the sorted labels in order from highest to lowest
print(list_index) 

# Print the first five predictions
for i in range(5):
  print(classification[list_index[i]], ':', round(prediction[0][list_index[i]] * 100, 2), '%' )

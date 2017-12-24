# Deep Columnar Convolutional Neural Network over MNIST dataset

# Importing the libraries
import keras
from keras.datasets import mnist
from keras.layers import merge, Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 10

# Input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Building the DCCNN architecture
# Initializing the network
ip = Input(shape = (1, img_rows, img_cols))

# Creating the forks for first layer
fork11 = Conv2D(32, 5, 5, activation = "relu", border_mode = "same")(input)
fork12 = Conv2D(32, 5, 5, activation = "relu", border_mode = "same")(input)
# Merging and pooling the forks for next layer
merge1 = merge([fork11, fork12], mode= "concat", concat_axis = 1, name = "merge1")
maxpool1 = MaxPooling2D(strides = (2,2), border_mode = "same")(merge1)

# Creating the forks for second layer with maxpool1 as input
fork21 = Conv2D(64, 4, 4, activation = "relu", border_mode = "same")(maxpool1)
fork22 = Conv2D(64, 4, 4, activation = "relu", border_mode = "same")(maxpool1)
# Merging and pooling the forks for next layer
merge2 = merge([fork21, fork22], mode= "concat", concat_axis = 1, name = "merge2")
maxpool2 = MaxPooling2D(strides = (2,2), border_mode = "same")(merge2)

# Creating last fork with maxpool2 as input
fork31 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
fork32 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
fork33 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
fork34 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
fork35 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
fork36 = Conv2D(128, 3, 3, activation = "relu", border_mode = "same")(maxpool2)
# Merging and pooling the forks for output
merge3 = merge([fork31, fork32, fork33, fork34, fork35, fork36], mode= "concat", concat_axis = 1, name = "merge3")
maxpool3 = MaxPooling2D(strides = (2,2), border_mode = "same")(merge3)

# Dropout layer
dropout = Dropout(0.5)(maxpool3)

# Flattening the layer
flatten = Flatten()(dropout)

# Creating a fully connected output layer
op = Dense(num_classes, activation = "softmax")(flatten)

model = Model(input = ip, output = op)
model.summary()

# Compiling the model generated
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Training over training data with test data as validation set
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Generating results
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



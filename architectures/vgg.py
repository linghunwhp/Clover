from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense


# https://github.com/amirhosseinzinati/Cifar10-VGG19--Tensorflow/blob/main/VGG19.ipynb
def vgg19(input_shape, num_classes=100):
    pre_model = VGG19(include_top=False, input_shape=input_shape)
    pre_model.trainable = True
    model = Sequential()
    model.add(pre_model)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

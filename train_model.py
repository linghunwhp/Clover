import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from architectures import vgg_old, Efficientnet, resnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule(epoch, start_lr=0.001):
    lr = start_lr
    if epoch == 200:
        lr = start_lr * 0.5e-3
    elif epoch == 150:
        lr = start_lr * 1e-3
    elif epoch == 100:
        lr = start_lr * 1e-2
    elif epoch == 50:
        lr = start_lr * 1e-1
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
    model_architecture = "efficientB7"  # [vgg16, lenet5, efficientB7, resnet20, resnet56]
    dataset_name = "svhn"   # [fashion_mnist, svhn, svhn, cifar10, cifar100]
    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        # hyper-parameters for training vgg16 with fashion_mnist dataset
        batch_size = 256
        epochs = 150
        data_augmentation = True
        num_classes = 10
        init_lr = 0.001

        with np.load('./data/fashion_mnist/fashion_mnist.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        input_shape = x_train.shape[1:]
        model = vgg_old.vgg16(input_shape=input_shape, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)

    elif model_architecture == "efficientB7" and dataset_name == "svhn":
        # hyper-parameters for training efficientB7 with svhn dataset
        batch_size = 256
        epochs = 150
        data_augmentation = True
        num_classes = 10
        init_lr = 0.001

        with np.load('./data/svhn/svhn.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        input_shape = x_train.shape[1:]
        model = Efficientnet.EfficientNetB7(input_shape=input_shape, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)

    elif model_architecture == "resnet20" and dataset_name == "cifar10":
        # hyper-parameters for training resnet-20 with cifar10 dataset
        batch_size = 256
        epochs = 150
        data_augmentation = True
        num_classes = 10
        depth = 20
        init_lr = 0.001

        with np.load('./data/cifar10/cifar10.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        input_shape = x_train.shape[1:]
        model = resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0, init_lr)), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)

    elif model_architecture == "resnet56" and dataset_name == "cifar100":
        # hyper-parameters for training resnet56 with cifar100 dataset
        batch_size = 256
        epochs = 200
        data_augmentation = True
        num_classes = 100
        depth = 56
        init_lr = 0.001

        with np.load('./data/cifar100/cifar100.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        input_shape = x_train.shape[1:]
        model = resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=init_lr), metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'checkpoint/' + dataset_name + '_' + model_architecture + '/saved_models/')
    model_name = dataset_name + '_' + model_architecture + '_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=20, min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, verbose=2, callbacks=callbacks)
    else:
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(x_train)
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_test, y_test), epochs=epochs, verbose=2, callbacks=callbacks, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=True)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

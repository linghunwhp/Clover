import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def lr_schedule_retrain(epoch):
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


def select(values, n, s='best', k=4):
    """
    n: the number of selected test cases.
    s: strategy, ['best', 'random', 'kmst', 'gini']
    k: for KM-ST, the number of ranges.
    """
    ranks = np.argsort(values)
    print(len(values), len(ranks))

    if s == 'best':
        h = n // 2
        return np.concatenate((ranks[:h], ranks[-h:]))

    elif s == 'random':
        return np.array(random.sample(list(ranks), n))

    elif s == 'kmst':
        fol_max = values.max()
        th = fol_max / k
        section_nums = n // k
        indexes = []
        for i in range(k):
            section_indexes = np.intersect1d(np.where(values < th * (i + 1)), np.where(values >= th * i))
            if section_nums < len(section_indexes):
                index = random.sample(list(section_indexes), section_nums)
                indexes.append(index)
            else:
                indexes.append(section_indexes)
                index = random.sample(list(ranks), section_nums - len(section_indexes))
                indexes.append(index)
        return np.concatenate(np.array(indexes))

    # This is for gini strategy. There is little difference from DeepGini paper. See function ginis() in metrics.py
    elif s == 'gini':
        return ranks[:n]


def sampling_selection(y_all, contextual_values, selected_num, selection_strategy="contextual_values", classes=10, by_class=True):
    print("*" * 10 + f"Selection Parameters: selected_num: {selected_num}, selection_strategy: {selection_strategy}" + "*" * 10)
    print(selected_num)
    if selection_strategy == "contextual_values":
        argsort_res = np.argsort(contextual_values)[::-1]
        if by_class:
            selected_index = []
            a_sorted_y_all = y_all[argsort_res]
            for c in range(classes):
                cur_select = 0
                for cur_y, cur_a_idx in zip(a_sorted_y_all, argsort_res):
                    if np.argmax(cur_y) == c:
                        cur_select += 1
                        selected_index.append(cur_a_idx)
                    if cur_select == int(selected_num / classes):
                        break
        else:
            selected_index = argsort_res[0:selected_num]
    elif selection_strategy == 'random':
        selected_index = np.array(random.sample(list(range(len(y_all))), selected_num))

    return selected_index


if __name__ == '__main__':
    model_architecture = "efficientB7"  # [vgg16, efficientB7, resnet20, resnet56]
    dataset_name = "svhn"   # [fashion_mnist, svhn, cifar10, cifar100]
    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        num_classes = 10
        train_data_path = './data/fashion_mnist/fashion_mnist.npz'
        ae_data_path = './checkpoint/fashion_mnist_vgg16/ae/'
        original_model_path = './checkpoint/fashion_mnist_vgg16/saved_models/fashion_mnist_vgg19_model.h5'

    elif model_architecture == "lenet5" and dataset_name == "svhn":
        num_classes = 10
        train_data_path = './data/svhn/svhn.npz'
        ae_data_path = './checkpoint/svhn_lenet5/ae/'
        original_model_path = './checkpoint/svhn_lenet5/saved_models/svhn_lenet5_model.h5'

    elif model_architecture == "efficientB7" and dataset_name == "svhn":
        num_classes = 10
        train_data_path = './data/svhn/svhn.npz'
        ae_data_path = './checkpoint/svhn_efficientB7/ae/'
        original_model_path = './checkpoint/svhn_efficientB7/saved_models/svhn_efficientB7_model.h5'

    elif model_architecture == "resnet20" and dataset_name == "cifar10":
        num_classes = 10
        train_data_path = './data/cifar10/cifar10.npz'
        ae_data_path = './checkpoint/cifar10_resnet20/ae/'
        original_model_path = './checkpoint/cifar10_resnet20/saved_models/cifar10_resnet20_model.h5'

    elif model_architecture == "resnet56" and dataset_name == "cifar100":
        num_classes = 100
        train_data_path = './data/cifar100/cifar100.npz'
        ae_data_path = './checkpoint/cifar100_resnet56/ae/'
        original_model_path = './checkpoint/cifar100_resnet56/saved_models/cifar100_resnet56_model.h5'

    with np.load(train_data_path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Load the generated adversarial inputs for training. FGSM and PGD.
    with np.load(ae_data_path + "/ae_fgsm_train.npz") as f:
        fgsm_train, fgsm_train_labels, fgsm_train_contextual_values = f['advs'], f['labels'], f['contextual_values']

    with np.load(ae_data_path + "/ae_pgd_train.npz") as f:
        pgd_train, pgd_train_labels, pgd_train_contextual_values = f['advs'], f['labels'], f['contextual_values']

    # Load the generated adversarial inputs for testing. FGSM and PGD.
    with np.load(ae_data_path + "/ae_fgsm_test.npz") as f:
        fgsm_test, fgsm_test_labels, fgsm_test_contextual_values = f['advs'], f['labels'], f['contextual_values']

    with np.load(ae_data_path + "/ae_pgd_test.npz") as f:
        pgd_test, pgd_test_labels, pgd_test_contextual_values = f['advs'], f['labels'], f['contextual_values']

    # Mix the adversarial inputs
    fp_train = np.concatenate((fgsm_train, pgd_train))
    fp_train_labels = np.concatenate((fgsm_train_labels, pgd_train_labels))
    fp_train_contextual_values = np.concatenate((fgsm_train_contextual_values, pgd_train_contextual_values))

    fp_test = np.concatenate((fgsm_test, pgd_test))
    fp_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))

    sNums = [int(len(fp_train)*0.01*i) for i in [1, 2, 4, 6, 8, 10, 20]]
    strategies = ['random', 'contextual_values']

    for num in sNums:
        print(num, round(num/len(fp_train), 2))
        for i in range(len(strategies)):
            strategy = strategies[i]
            model_path = "./checkpoint/" + dataset_name + "_" + model_architecture + "/retrain/best_Resnet_MIX_%d_%s.h5" % (num, strategy)
            checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
            lr_scheduler = LearningRateScheduler(lr_schedule_retrain)
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            callbacks = [checkpoint, lr_reducer, lr_scheduler]

            indexes = sampling_selection(y_all=fp_train_labels, contextual_values=fp_train_contextual_values, selected_num=num, selection_strategy=strategy, classes=num_classes, by_class=True)
            print(f"The # of selected test cases {len(indexes)}")
            selectAdvs = fp_train[indexes]
            selectAdvsLabels = fp_train_labels[indexes]

            x_train_mix = np.concatenate((x_train, selectAdvs), axis=0)
            y_train_mix = np.concatenate((y_train, selectAdvsLabels), axis=0)

            # load old model
            model = keras.models.load_model(original_model_path)
            datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            datagen.fit(x_train_mix)
            batch_size = 64
            history = model.fit_generator(datagen.flow(x_train_mix, y_train_mix, batch_size=batch_size), validation_data=(fp_test, fp_test_labels), epochs=40, verbose=2,
                                          callbacks=callbacks, steps_per_epoch=x_train_mix.shape[0] // batch_size, shuffle=True)

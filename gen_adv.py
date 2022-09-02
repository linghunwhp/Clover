import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from ae_utils.attack import FGSM, PGD
from tensorflow import keras
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    model_architecture = "vgg16"  # [vgg16, lenet5, efficientB7, resnet20, resnet56]
    dataset_name = "fashion_mnist"   # [fashion_mnist, svhn, svhn, cifar10, cifar100]
    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}")

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        num_classes = 10
        ep = 0.03
        train_data_path = './data/fashion_mnist/fashion_mnist.npz'
        model_path = './checkpoint/fashion_mnist_vgg16/saved_models/fashion_mnist_vgg19_model.h5'
        save_path = './checkpoint/fashion_mnist_vgg16/ae/'

    elif model_architecture == "lenet5" and dataset_name == "svhn":
        num_classes = 10
        ep = 0.03
        train_data_path = './data/svhn/svhn.npz'
        model_path = './checkpoint/svhn_lenet5/saved_models/svhn_lenet5_model.h5'
        save_path = './checkpoint/svhn_lenet5/ae/'

    elif model_architecture == "efficientB7" and dataset_name == "svhn":
        num_classes = 10
        ep = 0.03
        train_data_path = './data/svhn/svhn.npz'
        model_path = "./checkpoint/svhn_efficientB7/saved_models/svhn_efficientB7_model.078.h5"
        save_path = './checkpoint/svhn_efficientB7/ae/'

    elif model_architecture == "resnet20" and dataset_name == "cifar10":
        num_classes = 10
        ep = 0.01
        train_data_path = './data/cifar10/cifar10.npz'
        model_path = './checkpoint/cifar10_resnet20/saved_models/cifar10_resnet20_model.120.h5'
        save_path = './checkpoint/cifar10_resnet20/ae/'

    elif model_architecture == "resnet56" and dataset_name == "cifar100":
        num_classes = 100
        ep = 0.01
        train_data_path = './data/cifar100/cifar100.npz'
        model_path = './checkpoint/cifar100_resnet56/saved_models/cifar100_resnet56_model.086.h5'
        save_path = './checkpoint/cifar100_resnet56/ae/'

    with np.load(train_data_path) as f:
        x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    # preprocess cifar dataset
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model = keras.models.load_model(model_path)
    fgsm = FGSM(model, ep=ep, isRand=True)
    idxs_all = []
    advs_all = []
    labels_all = []
    contextual_values_all = []
    batch_size = 1000
    for i in range(round(len(x_train)/batch_size)):
        idxs, advs, labels, contextual_values = fgsm.generate(x_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size])
        idxs_all.extend(idxs+i*batch_size)
        advs_all.extend(advs)
        labels_all.extend(labels)
        contextual_values.extend(contextual_values)
    np.savez(save_path + '/adversarial_fgsm.npz', idxs=np.array(idxs_all), advs=np.array(advs_all), labels=np.array(labels_all), contextual_values=np.array(contextual_values_all))

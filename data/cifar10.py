import numpy as np
from tensorflow.keras.datasets import cifar10
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    np.savez('./cifar10/cifar10.npz', x_train=np.array(x_train), y_train=np.array(y_train), x_test=np.array(x_test), y_test=np.array(y_test))

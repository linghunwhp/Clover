import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)
    np.savez('./fashion_mnist/fashion_mnist.npz', x_train=np.array(x_train), y_train=np.array(y_train), x_test=np.array(x_test), y_test=np.array(y_test))

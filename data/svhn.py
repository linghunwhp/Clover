import os
import numpy as np
import scipy.io as sio


def extract_data(path, filename):
    data = sio.loadmat(os.path.join(path, filename))
    X = data['X'].transpose(3, 0, 1, 2)
    y = data['y'].reshape((-1))
    y[y == 10] = 0
    return X, y.astype(np.int)


def load_svhn(dir_path=None):
    # https: // www.programcreek.com / python /?CodeExample = load + svhn
    if os.path.exists(dir_path + "/svhn.npz"):
        f = np.load(dir_path + "/svhn.npz")
        return f['x_train'], f['y_train'], f['x_test'], f['y_test']
    else:
        x_train, y_train = extract_data(dir_path, 'train_32x32.mat')
        x_test, y_test = extract_data(dir_path, 'test_32x32.mat')
        np.savez(dir_path + '/svhn.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    load_svhn('./svhn/')

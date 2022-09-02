import numpy as np
import tensorflow as tf
from tensorflow import keras


class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=False):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        target = tf.constant(y, dtype=float)

        xi = x.copy()
        idxs_1 = np.where(np.argmax(self.model(xi), axis=1) == np.argmax(y, axis=1))[0]

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)

        x = tf.Variable(x, dtype=float)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x))
            grads = tape.gradient(loss, x)
        delta = tf.sign(grads)

        x_adv = x + self.ep * delta
        x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)

        x_adv_pred = self.model(x_adv)
        x_adv_pred_label = np.argmax(x_adv_pred, axis=1)
        idxs_2 = np.where(x_adv_pred_label != np.argmax(y, axis=1))[0]
        idxs = np.array(list(set(idxs_1).intersection(set(idxs_2))))
        x_adv, xi, target, x_adv_label, x_adv_confidence = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs], x_adv_pred_label[idxs], np.max(x_adv_pred, axis=1)

        contextual_value_list = []
        for original_x, ae, gt, ae_label, ae_pred_vec in zip(xi, x_adv, target, x_adv_label, x_adv_pred):
            peer_ae_array = []
            sample_num = 20
            for _ in range(sample_num):
                new_delta = self.ep * np.random.uniform(-1, 1, size=ae.shape)
                sample_adv = ae + new_delta
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=original_x - self.ep, clip_value_max=original_x + self.ep)
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=0, clip_value_max=1)
                peer_ae_array.append(sample_adv)

            peer_ae_array = np.array(peer_ae_array)
            pred = self.model(peer_ae_array)
            contextual_value_list.append(np.mean([tf.keras.losses.categorical_crossentropy(ae_pred_vec, vec).numpy() for vec in pred]))
        return np.array(idxs), x_adv.numpy(), target.numpy(), np.array(contextual_value_list)


class PGD:
    """
    We use PGD to generate a batch of adversarial examples. PGD could be seen as iterative version of FGSM.
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step == None:
            self.step = ep/6
        self.epochs = epochs
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        target = tf.constant(y, dtype=float)

        xi = x.copy()
        idxs_1 = np.where(np.argmax(self.model(xi), axis=1) == np.argmax(y, axis=1))[0]

        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)

        x_adv = tf.Variable(x, dtype=float)
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=xi-self.ep, clip_value_max=xi+self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
            x_adv = tf.Variable(x_adv)

        x_adv_pred = self.model(x_adv)
        x_adv_pred_label = np.argmax(x_adv_pred, axis=1)
        idxs_2 = np.where(x_adv_pred_label != np.argmax(y, axis=1))[0]
        idxs = np.array(list(set(idxs_1).intersection(set(idxs_2))))

        x_adv, xi, target, x_adv_label, x_adv_confidence = x_adv.numpy()[idxs], xi[idxs], target.numpy()[idxs], x_adv_pred_label[idxs], np.max(x_adv_pred, axis=1)

        contextual_value_list = []
        for original_x, ae, gt, ae_label, ae_pred_vec in zip(xi, x_adv, target, x_adv_label, x_adv_pred):
            peer_ae_array = []
            sample_num = 20
            for _ in range(sample_num):
                new_delta = self.ep * np.random.uniform(-1, 1, size=ae.shape)
                sample_adv = ae + new_delta
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=original_x - self.ep, clip_value_max=original_x + self.ep)
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=0, clip_value_max=1)
                peer_ae_array.append(sample_adv)

            peer_ae_array = np.array(peer_ae_array)
            pred = self.model(peer_ae_array)
            contextual_value_list.append(np.mean([tf.keras.losses.categorical_crossentropy(ae_pred_vec, vec).numpy() for vec in pred]))
        return np.array(idxs), x_adv.numpy(), target.numpy(), np.array(contextual_value_list)

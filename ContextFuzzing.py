import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from MeanShift import mean_shift

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Context:
    '''
    f is the model under testing
    s is the seed under fuzzing
    ground_truth is the ground truth of s
    s_gen is the generated test case from s
    delta is the radius to generate test cases surrounding s_gen
    k is the number of test cases to generate surrounding s_gen
    '''
    def __init__(self, f, s, ground_truth, s_gen, delta, k=10):
        self.f = f
        self.s = s
        self.ground_truth = ground_truth
        self.s_gen = s_gen
        self.delta = delta
        self.k = k
        self.test_suite = []
        self.prediction_matrix = []

    def materializes(self):
        for i in range(self.k):
            new_test_case = self.s_gen + np.random.uniform(-1*self.delta, 1*self.delta)
            new_test_case = tf.clip_by_value(new_test_case, clip_value_min=self.s_gen - self.delta, clip_value_max=self.s_gen + self.delta)
            new_test_case = tf.clip_by_value(new_test_case, clip_value_min=0, clip_value_max=1)
            self.test_suite.append(new_test_case)

    def predictions(self):
        model = tf.keras.models.load_model(f)
        for t in self.test_suite:
            self.prediction_matrix.append(model(t))

    def measure(self):
        model = tf.keras.models.load_model(f)
        s_pred_label = np.argmax(model(self.s))
        s_gen_pred_label = np.argmax(model(self.s_gen))
        if s_pred_label != self.ground_truth or (s_pred_label == self.ground_truth and s_gen_pred_label == self.ground_truth):
            return 0.0
        else:
            return np.mean(np.array(self.prediction_matrix)[:, s_gen_pred_label])


class ContextFuzzing:
    def __init__(self, model, max_step_size, steps=5, ep=0.01, bound=0.01, ctx_num=10, gctx_num=5):
        self.model = model
        self.max_step = max_step_size
        self.steps = steps
        self.ep = ep
        self.bound = bound
        self.ctx_num = ctx_num
        self.gctx_num = gctx_num

    def step(self, current_step):
        return self.max_step * np.sin(current_step*np.pi/(self.steps+1))

    def pre_generation(self, x, y):
        gradient_matrix = None
        target = tf.constant(y, dtype=float)
        xi = x.copy()
        idxs_1 = np.where(np.argmax(self.model(xi), axis=1) == np.argmax(y, axis=1))[0]
        x_adv = tf.Variable(x, dtype=float)
        for i in range(self.steps):
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            if i == 0:
                gradient_matrix = copy.deepcopy(delta.numpy())
            x_adv.assign_add(self.step(current_step=i) * delta * self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=xi - self.ep, clip_value_max=xi + self.ep)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
            x_adv = tf.Variable(x_adv)

        x_adv_pred = self.model(x_adv)
        x_adv_pred_label = np.argmax(x_adv_pred, axis=1)
        idxs_2 = np.where(x_adv_pred_label != np.argmax(y, axis=1))[0]
        idxs = np.array(list(set(idxs_1).intersection(set(idxs_2))))

        confidence_mean_list = []
        prediction_matrix = []
        for original_x, ae, gt, ae_label, ae_pred_vec in zip(xi, x_adv.numpy(), target.numpy(), x_adv_pred_label, x_adv_pred):
            peer_ae_array = []
            for _ in range(self.ctx_num):
                a = np.concatenate((np.ones(int(ae.shape[0] * ae.shape[1] * ae.shape[2] / 2)), -1 * np.ones(int(ae.shape[0] * ae.shape[1] * ae.shape[2] / 2))))
                np.random.shuffle(a)
                a = a.reshape(ae.shape)
                new_delta = self.ep * a
                # new_delta = self.ep * np.random.uniform(-1, 1, size=ae.shape)
                sample_adv = ae + new_delta
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=original_x - self.ep, clip_value_max=original_x + self.ep)
                sample_adv = tf.clip_by_value(sample_adv, clip_value_min=0, clip_value_max=1)
                peer_ae_array.append(sample_adv)

            peer_ae_array = np.array(peer_ae_array)
            pred = self.model(peer_ae_array)
            prediction_matrix.append(pred)
            confidence_mean_list.append(pred.numpy()[:, ae_label].mean())

        confidence_mean_list_array = np.array(confidence_mean_list)
        ac_array = np.array(copy.deepcopy(confidence_mean_list))
        ac_array[np.array(list(set(range(len(x)))-set(idxs)))] = 0

        return x_adv.numpy()[idxs], target.numpy()[idxs], confidence_mean_list_array[idxs], gradient_matrix, np.array(prediction_matrix), ac_array

    def fuzzing_process(self, x, y, pre_gen_target, pre_gen_advs, pre_gen_moc, grad_matrix, pred_matrix, ac_array):
        x_indexs = list(range(len(x)))
        adv_all = list(pre_gen_advs)
        label_all = list(pre_gen_target)
        contextual_value_all = list(pre_gen_moc)

        ac_dict = dict()
        g_dict = dict()
        for x_i in x_indexs:
            ac_dict[str(x_i)] = [ac_dict[x_i]]
            g_dict[str(x_i)] = [grad_matrix[x_i]]

        partition_labels = mean_shift(points=pred_matrix, h=2, MIN_DISTANCE=0.001)
        for x_index, img, label, pre_output, partition_label in zip(x_indexs, x, y, pred_matrix, partition_labels):
            similar_test_cases = [partition_idx for partition_idx in range(len(partition_labels)) if partition_labels[partition_idx]==partition_label]
            np.random.shuffle(similar_test_cases)
            gc_idxs = similar_test_cases[0:self.gctx_num]

            gc_grad = list(grad_matrix[gc_idxs])
            for gc_idx in gc_idxs:
                gc_grad.append(g_dict[str(gc_idx)])
            lc_grad = []

            target = label.reshape(-1, label.shape[0])
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            img_copy = img.copy()
            pred_label = np.argmax(self.model(img_copy))
            x_adv = tf.Variable(img, dtype=float)
            for i in range(len(gc_idxs)):
                with tf.GradientTape() as tape:
                    loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                    grads = tape.gradient(loss, x_adv)
                delta = tf.sign(grads)

                gc_delta = 0.0 if len(gc_grad) == 0 else np.mean(gc_grad, axis=0)
                lc_delta = 0.0 if len(lc_grad) == 0 else np.mean(lc_grad, axis=0)
                first_grad = delta + gc_delta + lc_delta
                x_adv.assign_add(self.max_step * first_grad * self.ep)

                for j in range(1, self.steps+1):
                    with tf.GradientTape() as tape:
                        loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                        grads = tape.gradient(loss, x_adv)
                    delta_new = tf.sign(grads)
                    x_adv.assign_add(self.step(current_step=j) * delta_new * self.ep)
                    x_adv = tf.clip_by_value(x_adv, clip_value_min=img_copy - self.ep, clip_value_max=img_copy + self.ep)
                    x_adv = tf.clip_by_value(x_adv, clip_value_min=0, clip_value_max=1)
                    x_adv = tf.Variable(x_adv)
                if len(gc_grad) > 0:
                    gc_grad.pop()

                ae_label = np.argmax(self.model(x_adv))
                if pred_label == np.argmax(label) and ae_label != np.argmax(label):
                    lc_grad.append(first_grad)
                    peer_ae_array = []
                    for _ in range(self.ctx_num):
                        a = np.concatenate((np.ones(int(img.shape[1] * img.shape[2] * img.shape[3] / 2)), -1 * np.ones(int(img.shape[1] * img.shape[2] * img.shape[3] / 2))))
                        np.random.shuffle(a)
                        a = a.reshape(img.shape)
                        new_delta = self.ep * a
                        # new_delta = self.ep * np.random.uniform(-1, 1, size=img.shape)
                        sample_adv = x_adv.numpy() + new_delta
                        sample_adv = tf.clip_by_value(sample_adv, clip_value_min=img_copy - self.ep, clip_value_max=img_copy + self.ep)
                        sample_adv = tf.clip_by_value(sample_adv, clip_value_min=0, clip_value_max=1)
                        peer_ae_array.extend(sample_adv)

                    peer_ae_array = np.array(peer_ae_array)
                    pred = self.model(peer_ae_array)
                    adv_all.append(x_adv.numpy())
                    label_all.append(label)
                    new_contextual_value = pred.numpy()[:, ae_label].mean()
                    contextual_value_all.append(new_contextual_value)
                    ac_dict[str(x_index)].append(new_contextual_value)

                    if new_contextual_value > np.mean(ac_dict[str(x_index)]):
                        g_dict[str(x_index)].append(first_grad)

        return np.array(adv_all), np.array(label_all), np.array(contextual_value_all)


if __name__ == "__main__":
    model_architecture = "efficientB7"  # [vgg16, lenet5, efficientB7, resnet20, resnet56]
    dataset_name = "svhn"   # [fashion_mnist, svhn, svhn, cifar10, cifar100]
    dataset_num = 3000
    print(f"Model architecture: {model_architecture}, dataset: {dataset_name}, dataset_num: {dataset_num}")

    if model_architecture == "vgg16" and dataset_name == "fashion_mnist":
        num_classes = 10
        ep = 0.03
        model_path = "./checkpoint/fashion_mnist_vgg16/saved_models/fashion_mnist_vgg19_model.064.h5"
        save_path = './checkpoint/fashion_mnist_vgg16/'
        with np.load('./data/fashion_mnist/fashion_mnist.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    elif model_architecture == "lenet5" and dataset_name == "svhn":
        num_classes = 10
        ep = 0.03
        model_path = "./checkpoint/svhn_lenet5/saved_models/svhn_lenet5_model.053.h5"
        save_path = './checkpoint/svhn_lenet5/'
        with np.load('./data/svhn/svhn.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    elif model_architecture == "efficientB7" and dataset_name == "svhn":
        num_classes = 10
        ep = 0.03
        model_path = "./checkpoint/svhn_efficientB7/saved_models/svhn_efficientB7_model.078.h5"
        save_path = './checkpoint/svhn_efficientB7/'
        with np.load('./data/svhn/svhn.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    elif model_architecture == "resnet20" and dataset_name == "cifar10":
        num_classes = 10
        ep = 0.01
        model_path = './checkpoint/cifar10_resnet20/saved_models/cifar10_resnet20_model.120.h5'
        save_path = './checkpoint/cifar10_resnet20/'
        with np.load('./data/cifar10/cifar10.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    elif model_architecture == "resnet56" and dataset_name == "cifar100":
        num_classes = 100
        ep = 0.01
        model_path = './checkpoint/cifar100_resnet56/saved_models/cifar100_resnet56_model.086.h5'
        save_path = './checkpoint/cifar100_resnet56/'
        with np.load('./data/cifar100/cifar100.npz') as f:
            x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

    # preprocess cifar dataset
    x_train = x_train.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    x_train = x_train[0:dataset_num]
    y_train = y_train[0:dataset_num]

    original_model = tf.keras.models.load_model(model_path)
    context_fuzzing = ContextFuzzing(model=original_model, max_step_size=1/5, steps=5, ep=ep, bound=ep, ctx_num=10, gctx_num=5)

    batch_size = 100
    all_adv = None
    all_target = None
    all_moc = None
    for i in range(int(np.ceil(len(x_train)/batch_size))):
        seeds = np.array(range(x_train.shape[0]))[i*batch_size: (i+1)*batch_size]
        images = x_train[seeds]
        labels = y_train[seeds]
        pre_advs, pre_target, pre_moc, pre_grads, pre_pred_matrix, ac = context_fuzzing.pre_generation(x=images, y=labels)
        cur_adv, cur_target, cur_moc = context_fuzzing.fuzzing_process(x=images, y=labels, pre_gen_target=pre_target, pre_gen_advs=pre_advs, pre_gen_moc=pre_moc, grad_matrix=pre_grads, pred_matrix=pre_pred_matrix, ac_array=ac)

        cur_adv = np.array([item.reshape(x_train.shape[1], x_train.shape[2], x_train.shape[3]) for item in cur_adv])
        cur_target = np.array([item for item in cur_target])
        cur_moc = np.array([item for item in cur_moc])

        if all_adv is None:
            all_adv = cur_adv
            all_target = cur_target
            all_moc = cur_moc
        else:
            all_adv = np.concatenate((all_adv, cur_adv))
            all_target = np.concatenate((all_target, cur_target))
            all_moc = np.concatenate((all_moc, cur_moc))

        print(f"Batch: {i}, Current length of total_sets: {len(all_adv)}")
        if (i+1)*batch_size == 100 or (i+1)*batch_size == 1000 or (i+1)*batch_size == 2000 or (i+1)*batch_size == 3000:
            np.savez(save_path + '/fuzzing/Context_Fuzzing/Context_Fuzz_' + str((i+1)*batch_size) + '.npz', advs=all_adv, labels=all_target, mocs=all_moc)


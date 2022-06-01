from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import idx2numpy
import pickle
import cv2
import os

np.random.seed(4)


class Layer:
    def __init__(self):
        self.in_mat = None
        self.out_mat = None

    def forward(self, in_mat):
        pass

    def backward(self, out_derv, alpha):
        pass


class Convolution(Layer):
    def __init__(self, fil_no, fil_dim, s, p, c=1):
        super().__init__()
        self.stride = s
        self.pad = p
        self.fil_no = fil_no
        self.fil_dim = fil_dim
        self.filters = np.random.randn(fil_no, c, fil_dim, fil_dim) * np.sqrt(2.0 / (fil_dim * fil_dim * c))
        self.biases = np.random.randn(fil_no, 1) * np.sqrt(2.0 / (fil_dim * fil_dim * c))
        # self.filters = np.arange(fil_no * c * fil_dim * fil_dim, dtype=float).reshape((fil_no, c, fil_dim, fil_dim))
        # self.biases = np.arange(fil_no, dtype=float).reshape((fil_no, 1))

    @staticmethod
    def padding_mat(in_mat, p):
        if p < 0:
            return in_mat[:, :, p:-p, p:-p]
        res = np.pad(in_mat, [(0, 0), (0, 0), (p, p), (p, p)])
        return res

    def forward(self, in_mat):
        self.in_mat = in_mat
        mat = self.padding_mat(in_mat, self.pad)
        dim = (mat.shape[-1] - self.fil_dim) // self.stride + 1
        res = np.zeros((mat.shape[0], self.fil_no, dim, dim))
        for i in range(dim):
            for j in range(dim):
                for k in range(self.fil_no):
                    res[:, k, i, j] = np.sum(
                        mat[:, :, i * self.stride:i * self.stride + self.fil_dim,
                            j * self.stride:j * self.stride + self.fil_dim]
                        * self.filters[k], axis=(1, 2, 3)) + self.biases[k]
        return res

    def backward(self, out_derv, alpha):
        dim = (out_derv.shape[-1] - 1) * self.stride + 1
        new_out = np.zeros((out_derv.shape[0], out_derv.shape[1], dim, dim))
        new_out[:, :, 0::self.stride, 0::self.stride] = out_derv
        df = np.zeros(self.filters.shape)

        mat = self.padding_mat(self.in_mat, self.pad)
        for i in range(self.fil_dim):
            for j in range(self.fil_dim):
                for k in range(self.filters.shape[1]):
                    df[:, k, i, j] = np.average(np.sum(mat[:, [k], i:i + dim, j:j + dim] * new_out, axis=(2, 3)),
                                                axis=0)

        db = np.average(np.sum(out_derv, axis=(2, 3)), axis=0).reshape(-1, 1)

        padded_out = self.padding_mat(new_out, self.fil_dim - 1 - self.pad)
        new_filter = np.flip(self.filters, axis=(2, 3))
        dx = np.zeros(self.in_mat.shape)

        for i in range(self.in_mat.shape[2]):
            for j in range(self.in_mat.shape[3]):
                for k in range(self.in_mat.shape[1]):
                    dx[:, k, i, j] = np.sum(padded_out[:, :, i:i + self.fil_dim, j:j + self.fil_dim] * new_filter[:, k],
                                            axis=(1, 2, 3))

        self.filters -= alpha * df
        self.biases -= alpha * db
        return dx


class MaxPooling(Layer):
    def __init__(self, fil_dim, s):
        super().__init__()
        self.fil_dim = fil_dim
        self.stride = s

    def forward(self, in_mat):
        self.in_mat = in_mat
        dim = (in_mat.shape[-1] - self.fil_dim) // self.stride + 1
        res = np.zeros((in_mat.shape[0], in_mat.shape[1], dim, dim))

        for i in range(dim):
            for j in range(dim):
                mat_slice = in_mat[:, :, i * self.stride:i * self.stride + self.fil_dim,
                                   j * self.stride:j * self.stride + self.fil_dim]
                res[:, :, i, j] = np.max(mat_slice, axis=(2, 3))

        return res

    def backward(self, out_derv, alpha):

        dx = np.zeros(self.in_mat.shape)
        for n in range(self.in_mat.shape[0]):
            for c in range(self.in_mat.shape[1]):
                ix = 0
                for i in range(0, self.in_mat.shape[2]-self.fil_dim+1, self.stride):
                    iy = 0
                    for j in range(0, self.in_mat.shape[3]-self.fil_dim+1, self.stride):
                        mat_slice = self.in_mat[n, c, i:i+self.fil_dim, j:j+self.fil_dim]
                        p, q = np.unravel_index(np.argmax(mat_slice), mat_slice.shape)
                        dx[n, c, i+p, j+q] = out_derv[n, c, ix, iy]
                        iy += 1
                    ix += 1
        return dx


class FullyConnected(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bias = np.random.randn(out_dim) * np.sqrt(2.0 / in_dim)
        self.filter = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)

    def forward(self, in_mat):
        self.in_mat = in_mat
        self.out_mat = np.matmul(self.in_mat, self.filter) + self.bias
        return self.out_mat

    def backward(self, out_derv, alpha):
        df = np.matmul(self.in_mat.T, out_derv) / out_derv.shape[0]
        db = np.average(out_derv, axis=0)
        output = np.matmul(out_derv, self.filter.T)
        self.filter -= alpha * df
        self.bias -= alpha * db

        return output


class Softmax(Layer):

    def forward(self, in_mat):
        self.in_mat = in_mat

        in_mat -= np.max(in_mat, axis=-1, keepdims=True)
        temp = np.exp(in_mat)  # exp(in_mat - max(in_mat, axis = -1)) ??

        # print(in_mat)
        s = np.sum(temp, axis=1, keepdims=True)
        s[s == 0.0] = 1
        self.out_mat = temp / s
        return self.out_mat

    def backward(self, out_derv, alpha):
        return self.out_mat - out_derv


class Activation(Layer):

    def forward(self, in_mat):
        self.in_mat = in_mat
        self.out_mat = in_mat
        self.out_mat[self.in_mat < 0.0] = 0.0
        return self.out_mat

    def backward(self, out_derv, alpha):
        temp = out_derv
        temp[self.in_mat < 0.0] = 0.0
        return temp


class Flatten(Layer):

    def forward(self, in_mat):
        self.in_mat = in_mat
        self.out_mat = np.array(in_mat).reshape(in_mat.shape[0], -1)
        return self.out_mat

    def backward(self, out_derv, alpha):
        return np.reshape(out_derv, self.in_mat.shape)


def new_log(x):
    x[x == 0.0] = 1.0
    return np.log(x)


def get_architecture(s):
    arr = []
    with open(s) as f:
        lines = f.readlines()
    f.close()

    commands = len(lines)
    lines = [line.strip() for line in lines]
    for i in range(commands):
        r = lines[i].split(' ')
        arr.append(r)

    # arr = np.array(arr)
    for i in range(commands):
        length = len(arr[i])
        for j in range(1, length):
            arr[i][j] = int(arr[i][j])

    return arr, commands


def get_network(ar, cmd, dim):
    ly = []
    init_dim = dim
    c = 0
    if dim == 28:
        c = 1
    elif dim == 32:
        c = 3
    for i in range(cmd):
        if ar[i][0] == "Conv":
            layer = Convolution(ar[i][1], ar[i][2], ar[i][3], ar[i][4], c)
            ly.append(layer)
            init_dim = (init_dim - ar[i][2] + 2 * ar[i][4]) // ar[i][3] + 1
            c = ar[i][1]
        elif ar[i][0] == "ReLU":
            layer = Activation()
            ly.append(layer)
        elif ar[i][0] == "Pool":
            layer = MaxPooling(ar[i][1], ar[i][2])
            ly.append(layer)
            init_dim = (init_dim - ar[i][1]) // ar[i][2] + 1
        elif ar[i][0] == "Flat":
            layer = Flatten()
            ly.append(layer)
            init_dim = init_dim * init_dim * c
        elif ar[i][0] == "FC":
            out_dim = ar[i][1]
            layer = FullyConnected(init_dim, out_dim)
            ly.append(layer)
            init_dim = out_dim
        elif ar[i][0] == "Softmax":
            layer = Softmax()
            ly.append(layer)
        else:
            print("\nWrong input\n")
    return ly


def get_mnist_data():
    file = './MNIST_dataset/train-labels.idx1-ubyte'
    train_label = idx2numpy.convert_from_file(file)
    train_label = np.array(train_label).reshape(-1, 1)

    file = './MNIST_dataset/t10k-labels.idx1-ubyte'
    test_label = idx2numpy.convert_from_file(file)
    test_label = np.array(test_label).reshape(-1, 1)

    file = './MNIST_dataset/train-images.idx3-ubyte'
    train_arr = idx2numpy.convert_from_file(file)
    # cv2.imshow("Image", train_arr[0])
    # cv2.waitKey(0)
    train_img = []
    train_img_no = train_arr.shape[0]
    for i in range(train_img_no):
        train_img.append(np.expand_dims(train_arr[i], axis=0))
    train_img = np.array(train_img)
    # print(train_img[0])                          #(60000, 1, 28, 28)

    file = './MNIST_dataset/t10k-images.idx3-ubyte'
    test_arr = idx2numpy.convert_from_file(file)

    test_img = []
    test_img_no = test_arr.shape[0]
    for i in range(test_img_no):
        test_img.append(np.expand_dims(test_arr[i], axis=0))
    test_img = np.array(test_img)
    # print(test_img.shape)                          #(10000, 1, 28, 28)

    shuffled_test = np.random.permutation(test_img.shape[0])
    validation_idx = shuffled_test[:5000]
    test_idx = shuffled_test[5000:]
    validation_img = test_img[validation_idx]
    validation_label = test_label[validation_idx]
    test_img = test_img[test_idx]
    test_label = test_label[test_idx]

    # print(validation_img.shape)
    # print(validation_label.shape)
    # print(test_img.shape)
    # print(test_label.shape)
    # print(np.arr(train_arr).shape)                       # (60000, 28, 28)
    # print(train_label.shape)                             # (60000, 1)
    # print(np.array(test_arr).shape)                      # (10000, 28, 28)
    # print(test_label.shape)                              # (10000, 1)
    # print(validation_label[0])

    return train_img, train_label, validation_img, validation_label, test_img, test_label
    # return {
    #     'train_img': train_img,
    #     'train_label': train_label,
    #     'validation_img': validation_img,
    #     'validation_label': validation_label,
    #     'test_img': test_img,
    #     'test_label': test_label
    # }


def shuffle(a, b):
    index = np.random.permutation(len(a))
    return np.array(a)[index], np.array(b)[index]


def read_cifar_10_file(s):
    with open(s, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')

    ls = list(dict1.values())
    batch_labels = ls[1]  # 10000
    batch_data = ls[2]  # 10000, 3072

    return batch_data, batch_labels


def get_cifar_10_data():
    batch_data1, batch_labels1 = read_cifar_10_file('Cifar_10_dataset/data_batch_1')
    batch_data2, batch_labels2 = read_cifar_10_file('Cifar_10_dataset/data_batch_2')
    batch_data3, batch_labels3 = read_cifar_10_file('Cifar_10_dataset/data_batch_3')
    batch_data4, batch_labels4 = read_cifar_10_file('Cifar_10_dataset/data_batch_4')
    batch_data5, batch_labels5 = read_cifar_10_file('Cifar_10_dataset/data_batch_5')

    test_batch, test_label = read_cifar_10_file('Cifar_10_dataset/test_batch')

    train_batches = [batch_data1, batch_data2, batch_data3, batch_data4, batch_data5]
    train_labels = [batch_labels1, batch_labels2, batch_labels3, batch_labels4, batch_labels5]

    # with open('Cifar_10_dataset/batches.meta', 'rb') as fo:
    #     dict1 = pickle.load(fo)
    # ls = list(dict1.values())
    # label_names = ls[1]
    # print(label_names)
    train_arr = np.concatenate(np.array(train_batches), axis=0).reshape((-1, 3, 32, 32))
    train_label = np.concatenate(np.array(train_labels), axis=0)

    train_arr, train_label = shuffle(train_arr, train_label)

    test_batch = np.array(test_batch).reshape((-1, 3, 32, 32))
    test_batch, test_label = shuffle(test_batch, test_label)

    validation_arr = np.array(test_batch[:5000])
    validation_label = np.array(test_label[:5000])
    test_arr = np.array(test_batch[5000:])
    test_label = np.array(test_label[5000:])

    # print(test_label[0:50])
    return train_arr, train_label, validation_arr, validation_label, test_arr, test_label


def train_toy_dataset(s):
    arr = []
    with open(s) as f:
        lines = f.readlines()
    f.close()
    commands = len(lines)
    lines = [line.strip() for line in lines]
    for i in range(commands):
        r = lines[i].split(' ')
        arr.append(r)

    for i in range(commands):
        length = len(arr[i])
        for j in range(length):
            arr[i][j] = int(arr[i][j])

    arr = np.array(arr)
    # print(arr.shape)
    x_arr = arr[:, 0:4]
    y_arr = arr[:, 4]
    # print(x_arr)
    return x_arr, y_arr


def train_toy_dataset2(s):
    df = pd.read_csv(s, sep='\t', header=None, engine='python')
    # col_no = df.columns.size
    # df[df.columns[col_no-1]] = df[df.columns[col_no-1]].astype(int)
    # print(df)
    df = np.array(df)
    x_arr = df[:, 0:df.shape[1] - 1]
    y_arr = df[:, df.shape[1] - 1].astype(int)
    return x_arr, y_arr
    # print(y_arr)


def one_hot_encode(a):
    # b = np.zeros((len(a), max(a) + 1))
    # b[np.arange(len(a)), a] = 1
    # return b

    onehot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = onehot_encoder.fit_transform(a.reshape(-1, 1)).astype(int)
    # print(one_hot_encoded.shape)
    return one_hot_encoded


def forward_propagation(layers, nl, x_arr):
    output = x_arr
    for j in range(nl):
        output = layers[j].forward(output)
    return output


def backward_propagation(layers, nl, y_arr, alpha=0.1):
    output = y_arr

    for j in reversed(range(nl)):
        output = layers[j].backward(output, alpha)
    return output


def cross_entropy_loss(y_cap, y_arr):
    return - np.average(np.sum(y_arr * new_log(y_cap), axis=1))
    # y_cap = new_log(y_cap)
    #
    # new_p = []
    # for i in range(b_size):
    #     new_p.append(y_arr[i].ravel().dot(y_cap[i].ravel()))
    # new_p = np.array(new_p)
    # return (-1) * np.sum(new_p) / b_size


def report_measurements(y_real, y_found):
    ls = cross_entropy_loss(y_found, y_real)
    y_real = np.argmax(y_real, axis=1).ravel()
    y_found = np.argmax(y_found, axis=1).ravel()

    acc = accuracy_score(y_real, y_found)
    f1 = f1_score(y_real, y_found, average='macro')

    # print('\nAccuracy: {:.2f}\n'.format(acc))
    # print('Macro F1-score: {:.2f}'.format(macro_f1))
    # print('Micro F1-score: {:.2f}\n'.format(f1_score(y_real, y_found, average='micro')))
    # print('Macro Recall: {:.2f}'.format(recall_score(y_real, y_found, average='macro')))
    # print('Micro Recall: {:.2f}\n'.format(recall_score(y_real, y_found, average='micro')))
    # print('Macro Precision: {:.2f}'.format(precision_score(y_real, y_found, average='macro', zero_division=True)))
    # print('Micro Precision: {:.2f}\n'.format(precision_score(y_real, y_found, average='micro', zero_division=True)))
    return ls, acc, f1


def plotter(m_train, m_val, title):
    directory = 'graphs'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    plt.plot(range(len(m_train)), m_train, 'r-')
    plt.plot(range(len(m_val)), m_val, 'b-')
    plt.title(title)
    plt.savefig(os.path.join(directory, f'{title}.png'))
    plt.close()


def train_model(layers, nl, x_arr, y_arr, x_valid, y_valid, epoch_no=1, b_size=500, alpha=0.001):
    loss_list = []
    acc_list = []
    f1_list = []
    loss_list_train = []
    acc_list_train = []
    f1_list_train = []
    x = None
    y = None
    for e in range(epoch_no):
        length = x_arr.shape[0]

        for i in range(0, length, b_size):
            x = x_arr[i: i + b_size]
            y = y_arr[i: i + b_size]
            forward_propagation(layers, nl, x)
            backward_propagation(layers, nl, y, alpha)

        output = forward_propagation(layers, nl, x_valid)
        a, b, c = report_measurements(y_valid, output)
        print('\nValidation')
        print('Loss: {:.2f}'.format(a))
        print('Accuracy: {:.2f}'.format(b))
        print('Macro F1-score: {:.2f}'.format(c))

        loss_list.append(a)
        acc_list.append(b)
        f1_list.append(c)

        output = forward_propagation(layers, nl, x)
        a, b, c = report_measurements(y, output)
        print('\nTrain')
        print('Loss: {:.2f}'.format(a))
        print('Accuracy: {:.2f}'.format(b))
        print('Macro F1-score: {:.2f}'.format(c))
        loss_list_train.append(a)
        acc_list_train.append(b)
        f1_list_train.append(c)

    plotter(loss_list_train, loss_list, 'loss')
    plotter(acc_list_train, acc_list, 'accuracy')
    plotter(f1_list_train, f1_list, 'f1 score')


def main():
    # c = Convolution(6, 3, 2, 2, 3)
    # y = c.forward(np.arange(2 * 3 * 21 * 21, dtype=float).reshape((2, 3, 21, 21)))
    # print(y.shape)
    # print(y)
    # p = MaxPooling(2, 2)
    # y = p.forward(np.arange(2*5*6*6, dtype=float).reshape((2, 5, 6, 6)))
    # print(y.shape)
    # print(y)
    # dx = p.backward(y, 0.1)
    # print(dx.shape)
    # print(dx)
    # soft = Softmax()
    # y = soft.forward(np.arange(5*4, dtype=float).reshape(5, 4))
    # print(y.shape)
    # print(y)
    # y_true = np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 0, 1, 0],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0],
    # ])
    # print(cross_entropy_loss(y, y_true))

    archi, no_of_commands = get_architecture("input.txt")

    x_train, y_train, x_valid, y_valid, x_tst, y_tst = get_mnist_data()              # mnist
    # x_train, y_train, x_valid, y_valid, x_tst, y_tst = get_cifar_10_data()             # cifar10

    # x_train, y_train = train_toy_dataset("./Test_Red/trainNN.txt")                 # toy1
    # x_tst, y_tst = train_toy_dataset("./Test_Red/testNN.txt")
    # -----------------------------------------------------------------------------------------------------

    # x_train, y_train = train_toy_dataset2("./Toy_Dataset/trainNN.txt")             # toy2
    # x_tst, y_tst = train_toy_dataset2("./Toy_Dataset/testNN.txt")
    # -----------------------------------------------------------------------------------------------------
    index = [i for i in range(len(y_train)) if y_train[i] in (0, 1)]
    x_train = x_train[index]
    y_train = y_train[index]
    x_train = x_train[:500]
    y_train = y_train[:500]
    index = [i for i in range(len(y_tst)) if y_tst[i] in (0, 1)]
    x_tst = x_tst[index]
    y_tst = y_tst[index]
    index = [i for i in range(len(y_valid)) if y_valid[i] in (0, 1)]
    x_valid = x_valid[index]
    y_valid = y_valid[index]

    x_train = x_train.astype(float) / np.max(x_train)
    y_train = one_hot_encode(y_train)
    x_tst = x_tst.astype(float) / np.max(x_tst)
    y_tst = one_hot_encode(y_tst)
    x_valid = x_valid.astype(float) / np.max(x_valid)
    y_valid = one_hot_encode(y_valid)

    layers_list = get_network(archi, no_of_commands, x_train.shape[-1])
    train_model(layers_list, no_of_commands, x_train, y_train, x_valid, y_valid)
    # train_model(layers_list, no_of_commands, x_train, y_train, x_tst, y_tst)

    # x_train = np.expand_dims(x_train, axis=2)
    # y_train = np.expand_dims(y_train, axis=2)                                # 500, 4, 1
    # print(y_train.shape)
    # x_tst = np.expand_dims(x_tst, axis=2)
    # y_tst = np.expand_dims(y_tst, axis=2)
    # print(y_tst.shape)


if __name__ == '__main__':
    main()

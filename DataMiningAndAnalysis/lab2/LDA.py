import json
import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from PCA import pca


label_path = 'C:\\Users\\sunsisi\\Desktop\\label'
data_path = 'C:\\Users\\sunsisi\\Desktop\\8_data'
save_path = 'C:\\Users\\sunsisi\\Desktop\\lda_data'
train_ids = []
test_ids = []


def LDA(X, y, k):
    label_ = list(set(y))
    X_classify = {}
    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    miu = np.mean(X, axis=0)
    miu_classify = {}
    for label in label_:
        miu1 = np.mean(X_classify[label], axis=0)
        miu_classify[label] = miu1

    # St = np.dot((X - mju).T, X - mju)
    # 计算类内散度矩阵Sw
    Sw = np.zeros((len(miu), len(miu)))
    for i in label_:
        Sw += np.dot((X_classify[i] - miu_classify[i]).T, X_classify[i] - miu_classify[i])

    # Sb = St-Sw
    # 计算类内散度矩阵Sb
    Sb = np.zeros((len(miu), len(miu)))
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((miu_classify[i] - miu).reshape(
            (len(miu), 1)), (miu_classify[i] - miu).reshape((1, len(miu))))

    # print(np.linalg.matrix_rank(Sw))
    # print(np.linalg.matrix_rank(Sb))

    # 计算S_w^{-1}S_b的特征值和特征矩阵
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    sorted_indices = np.argsort(eig_vals)
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]
    return topk_eig_vecs


def lda(train_x, train_y, test_x, k):
    f = LinearDiscriminantAnalysis(n_components=k)
    f.fit(train_x, train_y)
    test = f.transform(test_x)
    train = f.transform(train_x)
    return train, test


def load_train(data_path):
    x = []
    t = 'train'
    labels = os.listdir(os.path.join(data_path, t))
    global train_ids
    train_ids = labels
    for label in labels:
        img_paths = os.listdir(os.path.join(data_path, t, label))
        array = []
        for img_name in img_paths:
            img = cv2.imread(os.path.join(data_path, t, label, img_name), 0)
            img = cv2.resize(img, (32, 32))
            array.append(np.array(img).flatten())  # 将二维图像平铺为一维图像

        x.append(np.array(array).flatten())

    return np.array(x), get_y(labels, os.path.join(label_path, 'train'))


def load_test(data_path):
    x = []
    t = 'test'
    labels = os.listdir(os.path.join(data_path, t))
    global test_ids
    test_ids = labels
    for label in labels:
        img_paths = os.listdir(os.path.join(data_path, t, label))
        array = []
        for img_name in img_paths:
            img = cv2.imread(os.path.join(data_path, t, label, img_name), 0)
            img = cv2.resize(img, (32, 32))
            array.append(np.array(img).flatten())  # 将二维图像平铺为一维图像

        x.append(np.array(array).flatten())

    return np.array(x), get_y(labels, os.path.join(label_path, 'test'))


def get_y(labels, path):
    y = []
    for label in labels:
        with open(os.path.join(path, label+'.mp4.json'), 'r', encoding='utf-8') as fin:
            index = json.load(fin)
            y.append(index['final'][0])

    return y


def make_x_0_255(x):
    x = np.real(x)
    max_ = x.max()
    min_ = x.min()
    x = (x - min_) / (max_ - min_) * 256
    x = x.astype(np.uint8)
    return x


def save_data(train_x, test_x):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    if not os.path.exists(os.path.join(save_path, 'train')):
        os.makedirs(os.path.join(save_path, 'train'))
    if not os.path.exists(os.path.join(save_path, 'test')):
        os.makedirs(os.path.join(save_path, 'test'))

    # train_x = make_x_0_255(train_x)
    # test_x = make_x_0_255(test_x)

    k = 0
    for id in train_ids:
        np.savetxt(os.path.join(save_path, 'train', id), train_x[k], fmt="%.10f")
        k = k + 1

    k = 0
    for id in test_ids:
        np.savetxt(os.path.join(save_path, 'test', id), test_x[k], fmt="%.10f")
        k = k + 1


# 降维
if __name__ == '__main__':
    train_x, train_y = load_train(data_path)
    test_x, test_y = load_test(data_path)
    print(train_x.shape)

# method 1 #
    lda_train_x, lda_test_x = lda(train_x, train_y, test_x, 10)

# method 2 #
#     num = len(train_x)
#     x = np.vstack((train_x, test_x))
#     x = pca(x, 400)
#     x = np.real(x)
#     max_ = x.max()
#     min_ = x.min()
#     x = (x - min_) / (max_ - min_) * 256
#     x = x.astype(np.uint8)
#     train_x = x[:num]
#     test_x = x[num:len(x)]
#     w = LDA(test_x, test_y, 10)
#     lda_train_x = train_x.dot(w)
#     lda_test_x = test_x.dot(w)

    save_data(lda_train_x, lda_test_x)

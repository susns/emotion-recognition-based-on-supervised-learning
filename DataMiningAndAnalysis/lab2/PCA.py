import os
import cv2
import numpy as np


data_path = 'C:\\Users\\sunsisi\\Desktop\\8_data'
save_path = 'C:\\Users\\sunsisi\\Desktop\\pca_data'
types = []
ids = []
num = []


def pca(df, k):
    mean = np.mean(df, axis=0)
    new_df = df - mean
    print('mean')
    cov = np.cov(new_df, rowvar=0)
    print('cov')
    eigVals, eigVects = np.linalg.eig(cov)
    print('valvec')
    eigValIndic = np.argsort(-eigVals)
    print('sort')
    n_eigValIndic = eigValIndic[:k]
    n_eigVect = eigVects[:, n_eigValIndic]
    print('choose')
    data_ret = df.dot(n_eigVect)
    data_ret = np.array(data_ret)
    return data_ret, n_eigVect
    # return data_ret


def PCA(x, k):
    # 1、去平均值：按列求和作为该列代表维度的平均值，然后相减
    avg = np.sum(x, axis=0, keepdims=True)
    n = len(x)
    avg = avg/n
    x = np.subtract(x, avg)

    # 2、计算协方差、特征值、特征向量
    tx = np.transpose(x)
    S = np.matmul(tx, x)
    val, vec = np.linalg.eig(S)

    # 3、选择特征向量
    # sum_val = np.sum(val)
    index = np.argsort(-val)
    pic_val = val[index[:k]]
    pic_vec = vec[:, index[:k]]

    return np.matmul(x, pic_vec), pic_vec
    # return np.matmul(x, pic_vec)


def load_data():
    x = []
    global types, ids, num
    types = os.listdir(data_path)
    for t in types:
        labels = os.listdir(os.path.join(data_path, t))
        ids = ids + labels
        num.append(len(labels))
        for label in labels:
            img_paths = os.listdir(os.path.join(data_path, t, label))
            array = []
            for img_name in img_paths:
                img = cv2.imread(os.path.join(data_path, t, label, img_name), 0)
                img = cv2.resize(img, (32, 32))
                array.append(np.array(img).flatten())  # 将二维图像平铺为一维图像

            x.append(np.array(array).flatten())

    return np.array(x)


def save_data(x):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))

    x = np.real(x)
    max_ = x.max()
    min_ = x.min()
    x = (x - min_) / (max_ - min_) * 256
    x = x.astype(np.uint8)
    k = 0
    for t, n in zip(types, num):
        if not os.path.exists(os.path.join(save_path, t)):
            os.makedirs(os.path.join(save_path, t))
        for i in range(n):
            np.savetxt(os.path.join(save_path, t, ids[k]), x[k], fmt="%d")
            k = k + 1


# 降维
if __name__ == '__main__':
    x = load_data()
    print(x.shape)
    train_x = x[num[0]:len(x)]
    print(train_x.shape)
    new_train_x, w = pca(train_x, 400)
    x = x.dot(w)
    save_data(x)

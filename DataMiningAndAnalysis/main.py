import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def save_data():
    if not os.path.exists(os.path.join("C:\\Users\\sunsisi\\Desktop\\pca_data", '1')):
        os.makedirs(os.path.join("C:\\Users\\sunsisi\\Desktop\\pca_data", '1'))

    data = np.array([[1, 2],
                     [3, 4]])
    # 保存为整数
    np.savetxt('out.txt', data[1], fmt="%d")
    # 保存为2位小数的浮点数，用逗号分隔
    np.savetxt(os.path.join("C:\\Users\\sunsisi\\Desktop\\pca_data", '1', '2'), data, fmt="%.2f", delimiter=',')

    with open('out.txt') as f:
        for line in f:
            print(line, end='')


def pic_data():
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(a[0:5])  # 从下标0开始，取到下标（5-1）


def str_to_float(a):
    b = []
    for i in a:
        b.append(float(i))
    return b


title = 'test'
all = np.loadtxt(title + '.txt', delimiter=',')
print(all.shape)

x = all[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
label = all[:, 10]

plt.title(title)

f = TSNE(n_components=2, init='pca', random_state=0)
y = f.fit_transform(x)

plt.scatter(y[:, 0], y[:, 1], c=label)
plt.show()

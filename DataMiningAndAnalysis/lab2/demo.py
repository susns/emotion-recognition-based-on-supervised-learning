import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from PCA import PCA
from LDA import LDA


# 一个案例
if __name__ == '__main__':
    x = [
        [0.69, 0.49],
        [-1.31, -1.21],
        [0.39, 0.99],
        [0.09, 0.29],
        [1.29, 1.09],
        [0.49, 0.79],
        [0.19, -0.31],
        [-0.81, -0.81],
        [-0.31, -0.31],
        [-0.71, -1.01]
    ]
    x = np.array(x)
    y = [2, 3, 2, 1, 2, 2, 1, 3, 1, 3]
    y = np.array(y)

# 1 # ############### pca ################ #
    new_pca_x, pca_w = PCA(x, 1)
    plt.title('pca')

    # 降维前的散点
    plt.scatter(x[:, 0], x[:, 1], c=y)

    # pca 降维的直线
    m = np.linspace(-2, 2, 5)
    m.resize(1, 5)
    pca_line = pca_w.dot(m)
    plt.plot(pca_line[0], pca_line[1])

    # 降维后的散点
    pca_n = pca_w.dot(new_pca_x.transpose())
    plt.scatter(pca_n[0], pca_n[1], marker='*', c=y)

    # plt.gca().invert_yaxis()     #翻转y轴
    plt.title('pca')
    plt.axis("equal")
    plt.show()

# 2 # ############### lda ################ #
    lda_w = LDA(x, y, 1)
    new_lda_x = x.dot(lda_w)

    f = LinearDiscriminantAnalysis(n_components=1)
    f.fit(x, y)
    new_lda_x2 = f.transform(x)

    # 降维前散点
    plt.scatter(x[:, 0], x[:, 1], c=y)

    # lda 直线
    lda_line = lda_w.dot(m)
    plt.plot(lda_line[0], lda_line[1])

    # 降维后的散点
    lda_n = lda_w.dot(new_lda_x.transpose())
    plt.scatter(lda_n[0], lda_n[1], marker='*', c=y)

    plt.title('lda')
    plt.axis("equal")
    plt.show()

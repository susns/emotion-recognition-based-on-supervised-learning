import os
from Linear import load_data
from tool import visualize_confusion_matrix, load
from sklearn.svm import SVC, LinearSVC, NuSVC
import numpy as np
import matplotlib.pyplot as plt


def svm(kernel_kind, model_type, train_x, train_y, test_x, test_y):
    if model_type == 1:
        model = NuSVC(nu=0.39, kernel=kernel_kind)
    else:
        model = SVC(C=0.39, kernel=kernel_kind)

    model.fit(train_x, train_y)
    pred = model.predict(train_x)
    correct = np.sum(pred == train_y)
    print('训练准确率', correct / len(train_x))

    pred = model.predict(test_x)
    correct = np.sum(pred == test_y)
    print('测试准确率', correct / len(test_x))

    print()
    return correct / len(test_x)


def linearSVM(penalty_kind, c, train_x, train_y, test_x, test_y):
    if penalty_kind == 'l1':
        model = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-4,
                          C=c, multi_class='ovr', fit_intercept=True,
                          intercept_scaling=1, class_weight=None, verbose=0,
                          random_state=None, max_iter=4000)
    else:
        model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                          C=c, multi_class='ovr', fit_intercept=True,
                          intercept_scaling=1, class_weight=None, verbose=0,
                          random_state=None, max_iter=4000)

    model.fit(train_x, train_y)
    pred = model.predict(train_x)
    correct = np.sum(pred == train_y)
    print('训练准确率', correct / len(train_x))

    pred = model.predict(test_x)
    correct = np.sum(pred == test_y)
    print('测试准确率', correct / len(test_x))

    print()
    return correct / len(test_x)


if __name__ == '__main__':
    paths = ['pca_data', 'lda_data', 'my_lda_data']
    kinds = ['rbf', 'linear', 'poly', 'sigmoid']
    models = [1, 2]

# #- 1 -# 线性SVM
    # x = np.linspace(0.1, 1, 10)
    # # y = np.random.rand(6, 10)
    # y = []
    # penalty_kinds = ['l1', 'l2']
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    # for penalty_kind in penalty_kinds:
    #     for path in paths:
    #         train_x, train_y, test_x, test_y = load(path)
    #         yi = []
    #         for xi in x:
    #             print(penalty_kind, path, xi)
    #             yi.append(linearSVM(penalty_kind, xi, train_x, train_y, test_x, test_y))
    #         y.append(yi)
    #
    # plt.plot([], [], color='gray', label='l1')
    # plt.plot([], [], color='gray', label='l2', linestyle='--')
    #
    # plt.plot(x, y[0], color='blue', label=paths[0])
    # plt.plot(x, y[1], color='yellow', label=paths[1])
    # plt.plot(x, y[2], color='green', label=paths[2])
    # plt.scatter(x, y[0], color='blue', marker='o')
    # plt.scatter(x, y[1], color='yellow', marker='o')
    # plt.scatter(x, y[2], color='green', marker='o')
    #
    # plt.plot(x, y[3], color='blue', linestyle='--')
    # plt.plot(x, y[4], color='yellow', linestyle='--')
    # plt.plot(x, y[5], color='green', linestyle='--')
    # plt.scatter(x, y[3], color='blue', marker='^')
    # plt.scatter(x, y[4], color='yellow', marker='^')
    # plt.scatter(x, y[5], color='green', marker='^')
    # plt.title('线性SVM准确率')
    # plt.legend(loc='upper right')
    # plt.xlabel('惩罚系数')
    # plt.ylabel('准确率')
    # plt.show()

# #- 2 -# 非线性SVM
    y = []
    for model in models:
        for path in paths:
            yi = []
            train_x, train_y, test_x, test_y = load(path)
            for kind in kinds:
                print(model, path, kind)
                yi.append(svm(kind, model, train_x, train_y, test_x, test_y))
            y.append(yi)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    width = 0.2  # width of bar
    x = np.arange(3)
    y = np.array(y)

    fig, ax = plt.subplots(1, 2)
    data = np.array(y)[0:3]
    ax[0].bar(x, data[:, 0], width, color='#000080', label=kinds[0])
    ax[0].bar(x + width, data[:, 1], width, color='#0F52BA', label=kinds[1])
    ax[0].bar(x + (2 * width), data[:, 2], width, color='#6593F5', label=kinds[2])
    ax[0].bar(x + (3 * width), data[:, 3], width, color='#73C2FB', label=kinds[3])
    ax[0].set_ylim(0, 0.3)
    ax[0].set_ylabel('准确率')
    ax[0].set_xticks(x + width + width / 2)
    ax[0].set_xticklabels(paths)
    ax[0].set_xlabel('降维方法')
    ax[0].set_title('非线性SVM(NuSVC)准确率')
    ax[0].legend()

    # fig, ax = plt.subplots(1, 2, 2)
    data = np.array(y)[3:6]
    ax[1].bar(x, data[:, 0], width, color='#000080', label=kinds[0])
    ax[1].bar(x + width, data[:, 1], width, color='#0F52BA', label=kinds[1])
    ax[1].bar(x + (2 * width), data[:, 2], width, color='#6593F5', label=kinds[2])
    ax[1].bar(x + (3 * width), data[:, 3], width, color='#73C2FB', label=kinds[3])
    ax[1].set_ylim(0, 0.3)
    ax[1].set_xticks(x + width + width / 2)
    ax[1].set_xticklabels(paths)
    ax[1].set_xlabel('降维方法')
    ax[1].set_title('非线性SVM(SVC)准确率')
    ax[1].legend()

    plt.show()

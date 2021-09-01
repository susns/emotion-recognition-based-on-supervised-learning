import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from tool import visualize_confusion_matrix, visualize_confusion_matrix_, load_data


def linear(relative_path):
    print(relative_path)
    data_path = os.path.join('C:\\Users\\sunsisi\\Desktop', relative_path)
    label_path = 'C:\\Users\\sunsisi\\Desktop\\label'
    train_x, train_y, train_label = load_data(os.path.join(data_path, 'train'), os.path.join(label_path, 'train'))
    test_x, test_y, test_label = load_data(os.path.join(data_path, 'test'), os.path.join(label_path, 'test'))
    model = LogisticRegression(max_iter=5000)
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    correct = np.sum(pred == test_y)
    print('测试准确率', correct / len(test_x))
    if relative_path == 'pca_data':
        visualize_confusion_matrix_(pred, test_y, test_label)
    else:
        visualize_confusion_matrix(pred, test_y)

    pred = model.predict(train_x)
    correct = np.sum(pred == train_y)
    print('训练准确率', correct / len(train_x))
    visualize_confusion_matrix(pred, train_y)

    print()


if __name__ == '__main__':
    linear('pca_data')
    linear('lda_data')
    linear('my_lda_data')

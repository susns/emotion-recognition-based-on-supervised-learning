import torch
from torch import nn
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

from torch.autograd import Variable

validation = []
precisions = []
label_path = 'C:\\Users\\sunsisi\\Desktop\\label'
data_path = 'C:\\Users\\sunsisi\\Desktop\\8_data'
use_gpu = torch.cuda.is_available()


# 读取数据的函数,先读取标签，再读取图片
def _read(kind, size):
    x = []
    labels = os.listdir(os.path.join(data_path, kind))
    for label in labels:
        img_paths = os.listdir(os.path.join(data_path, kind, label))
        array = []
        for img_name in img_paths:
            img = cv2.imread(os.path.join(data_path, kind, label, img_name), 0)
            img = cv2.resize(img, (size, size))
            array.append(np.array(img))

        x.append(np.array(array))

    return np.array(x), get_y(labels, os.path.join(label_path, kind))


# 获取数据标签对应的类别
def get_y(labels, path):
    y = []
    for label in labels:
        with open(os.path.join(path, label + '.mp4.json'), 'r', encoding='utf-8') as fin:
            index = json.load(fin)
            y.append(index['final'][0])

    return np.array(y)


# 获取原数据8*size*size
def get_data(size):
    train_img, train_label = _read('train', size)
    test_img, test_label = _read('test', size)
    return [train_img, train_label, test_img, test_label]


# 获取降维后的数据
def load_data(data_path, label_path):
    x = []
    ids = os.listdir(data_path)
    for id in ids:
        array = np.loadtxt(os.path.join(data_path, id))
        x.append(array)

    return np.array(x), get_y(ids, label_path), ids


# 获取指定降维的数据
def load(path):
    data_path = os.path.join('C:\\Users\\sunsisi\\Desktop', path)
    label_path = 'C:\\Users\\sunsisi\\Desktop\\label'
    train_x, train_y, train_label = load_data(os.path.join(data_path, 'train'),
                                              os.path.join(label_path, 'train'))
    test_x, test_y, test_label = load_data(os.path.join(data_path, 'test'), os.path.join(label_path, 'test'))
    return train_x, train_y, test_x, test_y


# 训练函数
def train(epoch, model, train_loader, optimizer, criterion):
    # 调用前向传播
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(torch.int64)

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)  # 定义为Variable类型，能够调用autograd

        # 初始化时，要清空梯度
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # 相当于更新权重值


# 定义验证函数(用的训练集的随机样本验证，非验证集)
def validate(model, validate_loader, criterion):
    model.eval()  # 让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in validate_loader:
        target = target.to(torch.int64)

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        # 计算总的损失
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(validate_loader.dataset)
    validation.append(100. * correct / len(validate_loader.dataset))


# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()  # 让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.to(torch.int64)

        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        # 计算总的损失
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    precisions.append(100. * correct / len(test_loader.dataset))


# 精度可视化
def visualize_precision(iterations):
    x = np.linspace(1, iterations, iterations)
    plt.plot(x, validation, color='blue', label='training accuracy')
    plt.plot(x, precisions, color='yellow', label='testing accuracy')
    plt.legend()
    plt.title('the precision')
    plt.xlabel('iter')
    plt.ylabel('precision')
    plt.show()


# 可视化混淆矩阵
def visualize_confusion_matrix(pred, test_y):
    confusion_matrix = np.zeros(shape=(11, 11))
    for i, j in zip(test_y, pred):
        confusion_matrix[i][j] += 1
    print(confusion_matrix)


# 可视化混淆矩阵的同时，把原数据也分类出来
def visualize_confusion_matrix_(pred, test_y, labels):
    confusion_matrix = np.zeros(shape=(11, 11))
    label_names = ["anger", "disgust", "fear", "happiness", "neutral", "sadness",
                   "surprise", "contempt", "anxiety", "helplessness", "disappointment"]
    for i, j, label in zip(test_y, pred, labels):
        confusion_matrix[i][j] += 1
        shift(label, str(i) + label_names[i] + "\\" + str(j) + label_names[j])
    print(confusion_matrix)


# 把某个数据标签的八张人脸图，复制到另一个文件夹中
def shift(label, relative_path):
    path = 'C:\\Users\\sunsisi\\Desktop\\8_data\\test'
    new_path = 'C:\\Users\\sunsisi\\Desktop\\cm'

    path = os.path.join(path, label)
    new_path = os.path.join(new_path, relative_path)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    imgs = os.listdir(path)
    k = 0
    for img in imgs:
        file_path = os.path.join(path, img)
        new_file_path = os.path.join(new_path, label + '+' + str(k) + '.jpg')
        shutil.copy(file_path, new_file_path)
        k += 1


def get_precision(epoch):
    return precisions[epoch]


def get_validation(epoch):
    return validation[epoch]


# 从数据x,y中随机挑出m(80)个作为训练集的代表来测试模型准确度
def pick_m(x, y, m):
    indexes = np.random.choice(len(x), size=m, replace=False)
    pick_x = x[indexes]
    pick_y = y[indexes]
    return pick_x, pick_y

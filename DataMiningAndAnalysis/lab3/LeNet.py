import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tool import get_data, train, test, visualize_precision, get_precision, get_validation, pick_m, validate


def num_flat_features(x):
    # x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
    size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# 参数值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()


# 定义lenet5
class LeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        # 定义卷积层，8个输入通道，6个输通出道，5*5的卷积filter，因为输入的是32*32
        self.conv1 = nn.Conv2d(8, 6, 5, padding=0)
        # 第二个卷积层，6个输入，16个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 最后是三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        '''前向传播函数'''
        # 先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))

        # 第二次卷积+池化操作
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        # 重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, num_flat_features(x))

        # 三个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 定义一些超参数
    use_gpu = torch.cuda.is_available()
    batch_size = 20
    learning_rate = 0.0001
    iterations = 80
    kwargs = {'num_workers': 0, 'pin_memory': True}  # DataLoader的参数

    # 获取数据
    X, y, Xt, yt = get_data(32)
    train_x, train_y = torch.from_numpy(X.reshape(-1, 8, 32, 32)).float(), torch.from_numpy(y.astype(int))
    test_x, test_y = torch.from_numpy(Xt.reshape(-1, 8, 32, 32)).float(), torch.from_numpy(yt.astype(int))

    Xv, yv = pick_m(X, y, 80)
    validate_x, validate_y = torch.from_numpy(Xv.reshape(-1, 8, 32, 32)).float(), torch.from_numpy(yv.astype(int))

    # 封装好数据和标签
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    validate_dataset = TensorDataset(validate_x, validate_y)

    # 定义数据加载器
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    validate_loader = DataLoader(dataset=validate_dataset, shuffle=True, batch_size=batch_size, **kwargs)

    # 实例化网络
    model = LeNet5()
    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    # 定义代价函数，使用交叉熵验证
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 直接定义优化器，而不是调用backward
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # 调用参数初始化方法初始化网络参数
    model.apply(weight_init)

    # 调用函数执行训练和测试
    for epoch in range(iterations):

        train(epoch, model, train_loader, optimizer, criterion)
        validate(model, validate_loader, criterion)
        test(model, test_loader, criterion)

        print(epoch)
        print('验证', get_validation(epoch))
        print('测试', get_precision(epoch))

    visualize_precision(iterations)




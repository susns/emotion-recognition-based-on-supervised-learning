import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tool import get_data, train, test, visualize_precision, get_precision, get_validation, pick_m, validate


def Conv3x3BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def _make_layers(in_channels, out_channels, block_num):
    layers = [Conv3x3BNReLU(in_channels, out_channels)]
    for i in range(1, block_num):
        layers.append(Conv3x3BNReLU(out_channels, out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    def __init__(self, block_nums, num_classes=11):
        super(VGGNet, self).__init__()

        self.stage1 = _make_layers(in_channels=8, out_channels=64, block_num=block_nums[0])
        self.stage2 = _make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = _make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = _make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = _make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums)
    return model


def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGGNet(block_nums)
    return model


if __name__ == '__main__':
    # 定义一些超参数
    use_gpu = torch.cuda.is_available()
    torch.cuda.set_device(2)
    batch_size = 10
    learning_rate = 0.001
    iterations = 30
    kwargs = {'num_workers': 0, 'pin_memory': True}  # DataLoader的参数

    # 获取数据
    X, y, Xt, yt = get_data(224)
    train_x, train_y = torch.from_numpy(X.reshape(-1, 8, 224, 224)).float(), torch.from_numpy(y.astype(int))
    test_x, test_y = torch.from_numpy(Xt.reshape(-1, 8, 224, 224)).float(), torch.from_numpy(yt.astype(int))

    Xv, yv = pick_m(X, y, 80)
    validate_x, validate_y = torch.from_numpy(Xv.reshape(-1, 8, 224, 224)).float(), torch.from_numpy(yv.astype(int))

    # 封装好数据和标签
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    validate_dataset = TensorDataset(validate_x, validate_y)

    # 定义数据加载器
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    validate_loader = DataLoader(dataset=validate_dataset, shuffle=True, batch_size=batch_size, **kwargs)

    # 实例化网络
    model = VGG19()
    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    # 定义代价函数，使用交叉熵验证
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 直接定义优化器，而不是调用backward
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    # 调用函数执行训练和测试
    for epoch in range(iterations):
        train(epoch, model, train_loader, optimizer, criterion)
        validate(model, validate_loader, criterion)
        test(model, test_loader, criterion)

        print(epoch)
        print('验证', get_validation(epoch))
        print('测试', get_precision(epoch))

    visualize_precision(iterations)
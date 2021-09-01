import os
import matplotlib.pyplot as plt


# 挑出三人标注不同的数据
def selectDifferentDataFromAll():
    path_all = "D:\\DataMining\\data\\"
    file_diff = "C:\\Users\\sunsisi\\Desktop\\diffData.txt"
    path_diff = "C:\\Users\\sunsisi\\Desktop\\diffData\\"

    if not os.path.exists(path_diff):
        os.makedirs(path_diff)

    diff = open(file_diff, 'r')
    line = diff.readline()
    while line != "":
        arr = line.split(' ')
        file1 = path_all + arr[0] + ".mp4"
        file2 = path_diff + arr[0] + ".mp4"

        if not os.path.exists(file2):
            os.mknod(file2)

        origin = open(file1, 'rb')
        selected = open(file2, 'wb')
        selected.write(origin.read())
        selected.close()
        origin.close()
        line = diff.readline()

    diff.close()


# 定义函数来显示柱状上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.18, 1.01 * height, '%s' % int(height))


# 柱状图可视化类别分布
def generateAHistogram():
    file_path = "C:\\Users\\sunsisi\\Desktop\\allData.txt"
    f = open(file_path, 'r')

    y = {}
    x = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xlabel = ["invalid", "anger", "disgust", "fear", "happiness", "neutral", "sadness",
              "surprise", "contempt", "anxiety", "helplessness", "disappointment"]
    for i in x:
        y[i] = 0

    line = f.readline()
    while line != "":
        arr = line.split(' ')
        i = int(arr[1])
        y[i] += 1
        line = f.readline()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('视频情感分布柱状图')
    # ax.set_xlabel('情感')
    # ax.set_ylabel('数量')
    view = ax.bar(range(len(y)), y.values())
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ax.set_xticklabels(xlabel, rotation=20, fontsize='small')
    autolabel(view)
    plt.show()




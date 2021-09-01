import os
import matplotlib.pyplot as plt

data_path = 'C:\\Users\\sunsisi\\Desktop\\train_all'
save_path = 'C:\\Users\\sunsisi\\Desktop\\train_data'


# 找出路径里面人像图片不是8张的数据id
def select(path):
    res = []
    folders = os.listdir(path)
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder))
        if len(imgs) < 8:
            res.append(folder)

    print(len(res), res)


# 可视化某个路径下人脸图片个数的分布情况
def view_label_img_num(path):
    nums = []
    folders = os.listdir(path)
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder))
        nums.append(len(imgs))

    dic = {}
    max_ = max(nums)
    for i in range(max_):
        dic[i] = 0
    dic[max_] = 0

    for i in nums:
        dic[i] = dic[i]+1

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

    plt.title('剪裁视频得到的人脸分布情况')
    plt.xlabel('单个视频剪裁出来的人脸数')
    plt.ylabel('视频数量')
    plt.bar(dic.keys(), dic.values())
    plt.show()


# 删除路径下人脸图片少于某个特定值的样本
def delete_num_less_than(num=8):
    folders = os.listdir(data_path)
    for folder in folders:
        images = os.listdir(os.path.join(data_path, folder))
        if len(images) < num:
            for img in images:
                os.remove(os.path.join(data_path, folder, img))
            os.rmdir(os.path.join(data_path, folder))


# 为每个不少于8张人脸图片的样本均匀挑选8张
def pick_img_from_every_video(num=8):
    global data_path
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))

    folders = os.listdir(data_path)
    for folder in folders:
        images = os.listdir(os.path.join(data_path, folder))
        sum_ = len(images)
        step = int((sum_-1)/(num-1))
        images_pick = images[0:sum_:step]
        images_pick = images_pick[0:8]

        if not os.path.exists(os.path.join(save_path, folder)):
            os.makedirs(os.path.join(save_path, folder))

        for img in images_pick:
            fout = open(os.path.join(save_path, folder, img), 'wb')
            fin = open(os.path.join(data_path, folder, img), 'rb')
            fout.write(fin.read())
            fout.close()
            fin.close()


# select(data_path)
# delete_num_less_than()
pick_img_from_every_video()
# view_label_img_num(save_path)
# select(save_path)
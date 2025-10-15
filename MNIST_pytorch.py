# 导入库
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas
import matplotlib.pyplot as plt


# 构建神经网络类
class Classifier(nn.Module):

    def __init__(self):
        # 初始化pytorch父类
        super().__init__()

        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            # nn.LeakyReLU(0.02),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )

        # 创建损失函数
        self.loss_function = nn.MSELoss()
        # self.loss_function = nn.CrossEntropyLoss()
        # self.loss_function = nn.BCELoss()

        # 创建优化器（简单梯度下降）
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # 记录训练进展的计数器和列表
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        # 直接运行模型
        return self.model(inputs)

    def train(self, inputs, targets):
        # 计算网络的输出值
        outputs = self.forward(inputs)
        # 计算损失值
        loss = self.loss_function(outputs, targets)
        # 梯度归零，反向传播，并更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # 每隔10个训练样本增加一次计数器的值，并将损失值添加进列表末尾
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        if (self.counter % 1000 == 0):
            print("counter = ", self.counter)

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0,1.0), figsize=(16,8), alpha=0.1,
                marker='.', grid=True, yticks=(0,0.25,0.5,1.0,1.5))



# 创建一个MnistDataset类
class MnistDataset(Dataset):

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 目标图像（标签）
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0

        # 图像数据，取值范围为0~255,标准化为0~1
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].to_numpy()) / 255.0

        # 返回标签、图像数据张量以及目标张量
        return label,image_values,target

    def plot_image(self,index):
        arr = self.data_df.iloc[index,1:].to_numpy().reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(arr, interpolation='none', cmap='gray')

### 训练数据集
mnist_dataset = MnistDataset('data/mnist_train.csv')

# 创建神经网络
C = Classifier()

# 在MNITST数据集训练神经网络
epochs = 3
for i in range(epochs):
    print('training epoch', i+1, 'of', epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
### 绘制损失图
C.plot_progress()

mnist_test_dataset = MnistDataset('data/mnist_test.csv')

### 测试集上的表现
score = 0
items = 0

for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(image_data_tensor).detach().numpy()
    if (answer.argmax() == label):
        score += 1
    items += 1
print(score, items, score/items)
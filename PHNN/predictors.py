"""

这段代码定义了一个名为 MLP 的多层感知机（Multilayer Perceptron）类，它继承自 torch.nn.Module。MLP 类是一个简单的密集连接的前馈神经网络，可以作为预测器使用。以下是类的组成部分和它们的功能：

__init__(self, dense_layers, softmax = True): 构造函数接收一个列表 dense_layers，该列表定义了每一层的神经元数量。它还接收一个可选参数 softmax，用于指定是否在最后一层使用 softmax 激活函数。

self.dense_layers: 使用 nn.ModuleList 创建一个线性层的列表。每个线性层都是通过 nn.Linear 创建的，它将前一层的输出作为输入。

self.softmax: 一个布尔值，用于控制是否在最后一层使用 softmax 激活函数。

getLength(self): 一个方法，返回网络中线性层的数量。

forward(self, x): 前向传播函数，它接收输入数据 x 并通过网络层进行计算。输入数据首先被重塑为二维张量，然后逐层传递。在每个线性层之后，应用 softplus 激活函数。如果 softmax 参数为 True，则在最后一层使用 log_softmax 激活函数。

这个 MLP 类可以用于构建一个简单的前馈神经网络，它可以用于分类或其他预测任务。在训练过程中，你可以通过调整 dense_layers 参数来改变网络的深度和宽度，以及通过设置 softmax 参数来控制输出层的激活函数。
"""


import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dense_layers, softmax = True):
        """Simple dense MLP class used as predictor"""
        super().__init__()
        self.dense_layers = nn.ModuleList([nn.Linear(dense_layers[i], dense_layers[i + 1]) \
                                           for i in range(len(dense_layers) - 1)])
        self.softmax = softmax
        
    def getLength(self):
        return len(self.dense_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.dense_layers:
            l_x = l(x)
            x = F.softplus(l_x, beta=10,threshold=20)
        if not self.softmax: return l_x
        else: return F.log_softmax(l_x, dim=-1)
"""
这段代码定义了一个名为 genpoints 的函数，它用于在给定的二维区域内生成一个点网格。这个函数接受以下参数：

xmin, xmax：定义了点网格在 x 轴上的最小值和最大值。
ymin, ymax：定义了点网格在 y 轴上的最小值和最大值。
n_points：在 x 轴和 y 轴上各生成多少个点，因此总点数将是 n_points 的平方。
函数的工作流程如下：

使用 torch.linspace 在指定的 x 轴和 y 轴范围内生成等间隔的点。torch.linspace(xmin, xmax, n_points) 生成一个从 xmin 到 xmax 的一维张量，包含 n_points 个元素。

初始化一个空列表 p，用于存储生成的点。

通过两层嵌套循环遍历 x 轴和 y 轴上的点，将每个点的坐标组合成一个二维点，并添加到列表 p 中。

最后，使用 torch.Tensor(p) 将列表 p 转换为一个 PyTorch 张量，并返回该张量。

这个函数可以用来生成用于训练机器学习模型的合成数据集，或者用于任何需要在特定区域内生成均匀分布点的场景。

例如，如果你想在 x 轴范围 [-1, 1] 和 y 轴范围 [-1, 1] 内生成一个包含 100x100 个点的网格，你可以这样调用函数：
"""


import torch

def genpoints(xmin, xmax, ymin, ymax, n_points):
    """Generate a meshgrid contained in [xmin, xmax], [ymin, ymax] with `n_points` points"""
    xx = torch.linspace(xmin, xmax, n_points)
    yy = torch.linspace(ymin, ymax, n_points)
    p = []
    for i in range(n_points):
        for j in range(n_points):
            p.append([xx[i], yy[j]])
    return torch.Tensor(p)



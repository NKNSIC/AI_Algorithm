"""
这段代码是一个使用Port-Hamiltonian动力学优化的线性神经网络的低级实现。它包含了网络的定义、损失计算、梯度计算、批量处理以及训练和测试过程。以下是代码的主要组成部分：

simple_net(u, w): 定义了一个简单的线性神经网络，接受输入 u 和权重 w，并返回网络的输出。

loss(x, u, yh, a, b, c): 计算给定参数下的损失值，包括数据损失和正则化项。

gradient(x, u, yh, a, b, c): 计算损失函数的梯度。

hamiltonian_model(x, t, u, yh, beta, a, b, c): 定义了Port-Hamiltonian模型的常微分方程（ODE）。

loss_batch(bs, x, U, Yh, a, b, c): 计算批量数据的平均损失。

gradient_batch(bs, x, U, Yh, a, b, c): 计算批量数据的平均梯度。

ham_mod_batch(x, t, bs, U, Yh, beta, a, b, c): 定义了批量数据的Port-Hamiltonian模型的ODE。

train(X, y, bs, epochs, x0, a, b, c, beta, t): 训练过程，使用 odeint 函数来解决ODE，并在每个批次上迭代更新网络参数。

test(x, Xh, yh, trained, train_test): 测试过程，评估模型在测试数据上的准确性。

代码中使用了 numpy 和 scipy.integrate 库来处理数学运算和ODE求解，以及 tqdm 库来显示进度条。

要使用这段代码，你需要准备训练数据 X 和 y，设置批量大小 bs，初始化参数 x0，以及设置损失函数的超参数 a, b, c 和 beta。然后，你可以调用 train 函数来训练模型，并使用 test 函数来评估模型的性能。
"""


"""Low level implementation of a linear neural network optimized with Port-Hamiltonian
dynamics."""

import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm_notebook as tqdm

def simple_net(u, w):
    """Simple linear neural network"""
    # inputs
    u1, u2 = u[0], u[1]
    # weights of y1
    w11, w12, w13 = w[0], w[1], w[2]
    # weights of y2
    w21, w22, w23 = w[3], w[4], w[5]

    y1 = w11*u1 + w12*u2 + w13
    y2 = w21*u1 + w22*u2 + w23
    return np.array([y1,y2]).T


def loss(x, u, yh, a, b, c):
    """Computes loss J"""
    n = len(x) // 2
    w, dw = x[0:n], x[n:2*n]
    J = np.array(yh - simple_net(u, w))
    loss = a*J.dot(J) + b*dw.dot(dw) + c*w.dot(w)
    return loss


def gradient(x, u, yh, a, b, c):
    """Computes gradient of the loss J"""
    # x contains weights AND momenta
    n = len(x)//2
    # weights and their velocities
    w, dw = x[0:n], x[n:2*n]
    # inputs
    u1, u2 = u[0], u[1]
    # ground truth (reference outputs)
    yh1, yh2 = yh[0], yh[1]
    # weights of y1
    w11, w12, w13 = w[0], w[1], w[2]
    # weights of y2
    w21, w22, w23 = w[3], w[4], w[5]

    y = simple_net(u, w)
    # gradient computation
    dJ_w = 2.*a*np.array([u1*(y[0]-yh1), u2*(y[0]-yh1), y[0]-yh1, u1*(y[1]-yh2), u2*(y[1]-yh2), y[1]-yh2]).T
    #
    dJ_dw = np.array([2.*b*dw[0], 2.*b*dw[1], 2.*b*dw[2], 2.*b*dw[3], 2.*b*dw[4], 2.*b*dw[5]]).T
    dJ = np.hstack((dJ_w, dJ_dw)).T
    # regularisation term
    dJr = np.array([2.*c*w11, 2.*c*w12, 2.*c*w13, 2.*c*w21, 2.*c*w22, 2.*c*w23, 0., 0., 0., 0., 0., 0.])
    return dJ + dJr


def hamiltonian_model(x, t, u, yh, beta, a, b, c):
    """Defines ODE of the PH Model"""
    n = len(x)//2
    # Compute the gradient
    dJ = gradient(x, u, yh, a, b, c)
    # Compute derivative
    dwdt = dJ[n:2*n]/b
    ddwdt = -dJ[0:n] - beta*dJ[n:2*n]/b
    dxdt = np.hstack((dwdt, ddwdt))
    return dxdt


def loss_batch(bs, x, U, Yh, a, b, c):
    """Calculates loss J for a batch"""
    J_tot = 0
    for i in range(bs):
        u = U[i]
        yh = Yh[i]
        J_tot = J_tot + loss(x, u, yh, a, b, c)
    return J_tot/bs


def gradient_batch(bs, x, U, Yh, a, b, c):
    """Calculates gradient for a batch"""
    n = len(x) // 2
    dJ_tot = np.zeros((1, 2*n))
    dJ_tot = dJ_tot[0] 
    for i in range(bs):
        u = U[i]
        yh = Yh[i]
        dJ_tot = dJ_tot + gradient(x, u, yh, a, b, c)
    return dJ_tot/bs 


def ham_mod_batch(x, t, bs, U, Yh, beta, a, b, c):
    """Defines ODE of the PH Model (batch)"""
    n = len(x) // 2
    # define the matrix F
    On = np.zeros((n, n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    F = np.vstack((np.hstack((On, In)), np.hstack((-In, -B))))
    # Compute the gradient
    dJ = gradient_batch(bs, x, U, Yh, a, b, c)
    # Compute derivative
    dxdt = F.dot(dJ)
    return dxdt


def train(X, y, bs, epochs, x0, a, b, c, beta, t):
    """Trains a PHNN"""
    N_tot = len(X)
    N_batch = int(N_tot/bs)
    Ub = X.reshape(N_batch, bs, 2)
    Yhb = y.reshape(N_batch, bs, 2)
    # initialize variable to store results
    xf = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],x0], dtype='float')
    tf = np.array([0.])
    J = np.array([0.])
    xep = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],x0], dtype='float')
    Jep = loss_batch(bs, np.array(x0), Ub[0], Yhb[0], a, b, c)
    for j in tqdm(range(epochs)):
        for i in tqdm(range(N_batch)):
            U = Ub[i]
            Yh = Yhb[i]
            x0 = xf[-1] 
            sol = odeint(ham_mod_batch, x0, t, args=(bs, U, Yh, beta, a, b, c))
            xf = np.vstack((xf, sol))
            tf = np.hstack((tf, t+tf[-1]))
            N = len(sol)
            Ji = np.zeros((N,1))
            for i in range(N):
                Ji[i] = loss_batch(bs, sol[i], U, Yh, a, b, c)
            J = np.vstack((J, Ji))
        xep = np.vstack((xep, xf[-1]))
        Jep = np.vstack((Jep, J[-1]))
    xf = xf[2:-1]
    J = J[1:-1]
    tf = tf[1:-1]
    xep = xep[1:-1]
    return tf, xf, J, xep, Jep


def test(x, Xh, yh, trained, train_test):
    """Performs evaluation on out-of-sample data"""
    n = len(yh)
    count_predicted = 0
    for i in range(n):
        y = simple_net(Xh[i], x)
        if (y[0] > y[1]) and (yh[i, 0] > yh[i, 1]):
            count_predicted += 1
    accuracy = count_predicted*100./n
    #
    if train_test:
        if trained:
            print(f'Post-training train set accuracy: {accuracy} %')
        else:
            print(f'Pre-training train set accuracy: {accuracy} %')
    else:
        if trained:
            print(f'Post-training test accuracy: {accuracy} %')
        else:
            print(f'Pre-training test accuracy: {accuracy} %')
    return accuracy
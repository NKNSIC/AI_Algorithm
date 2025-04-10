"""
这是 PHNN 类的完整代码，它是一个用于模拟Port-Hamiltonian Differential Neural Networks（PHNN）的高级封装。这个类包含了初始化模型、创建状态向量、梯度计算、参数记录、前向传播以及训练过程的方法。下面是对关键方法的简要说明：

__init__: 类的构造函数，初始化模型参数和预测器。

createStateVector: 创建状态向量，包括权重和速度。

flattenParamVector: 将模型参数展平成一个向量，便于梯度计算和状态更新。

makeFMatrix: 创建用于参数状态动力学的F矩阵。

gradient: 计算梯度。

additionalTermsLoss: 计算损失函数的附加项，通常是正则化项。

assignNewState: 分配新的状态向量。

loadStateDict: 加载状态字典，更新模型参数。

flattenGradient: 将梯度展平成一个向量。

assignFlatGradient: 分配展平的梯度。

getConcatGradient: 获取连接的梯度向量。

fixInputOutput: 设置输入和输出数据。

setXi: 设置初始状态向量。

pred_accuracy: 计算模型在测试集上的准确率。

initializeRecord: 初始化记录损失和参数的列表。

recordLoss: 记录损失。

plotLoss: 绘制损失曲线。

recordParameters: 记录参数。

plotParameters: 绘制参数曲线。

plotVelocities: 绘制速度曲线。

forward: 定义模型的前向传播。

fit: 训练模型，使用 solve_ivp 来解决ODE。

这个类的使用需要一个数据加载器 trainloader，它应该是一个 DataLoader 对象，用于提供训练数据。训练过程中，模型会通过梯度下降和ODE求解来更新参数。

请注意，这个类依赖于 predictors.py 中的 MLP 类，以及 torch 和 scipy 等库。在使用之前，请确保所有依赖项都已正确安装，并且 predictors.py 在同一个目录下或者在Python的模块搜索路径中。此外，你可能需要根据你的具体需求调整模型的参数和训练过程。
"""




import sys
import numpy as np
from scipy.integrate import solve_ivp
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from predictors import MLP
import matplotlib.pyplot as plt


class PHNN(nn.Module):
    """
    High level wrapper for Port-Hamiltonian Differential Neural Networks
    
    :dense_layer: list of dimensions of linear layers of the chosen HDNN predictor \
    (e.g [12,24,2] for 12-dimensional inputs and output size 2)
    :p_type: imported nn.Module class with forward method to be used as predictor
    :p_args: arguments for p_type. List format expected, unpacked with *args
    :hparams: list [a,b,c] of loss function hyperparameters
    :beta: beta of F function of the weight dynamics
    :p_module: module name from which the predictor class is imported 
    """
    def __init__(self, p_type, p_args, hparams, beta, device, p_module=__name__):
      
        # initialize superclass method
        super().__init__()
        self.device = device

        # predictor initialization
        self.p_type = getattr(sys.modules[p_module], p_type)
        self.p_args = p_args
        
        self.predictor = self.p_type(*self.p_args).to(self.device)
        self.len = len(self.predictor.state_dict())
        self.lenp = sum(1 for _ in iter(self.predictor.parameters()))
               
        # simple assignment routine
        self.hparams = hparams
        self.dJ,self.dJddw = [], []
        self.beta = beta
           
        # creating flattened versions of self.w and self.wdot for gradient and state computations
        self.flat_w = self.flattenParamVector() 
        self.flat_wdot = torch.rand(self.flat_w.shape)
     
        self.count = 0
        
        #time counter for loss and parameter plotting
        self.time = 0 
        self.pLoss = []
        self.pW = []
        self.pWdot = []
        self.initializeRecord()
        
    def createStateVector(self,velocity=True,first_instance=True):
        w = []
        wdot = []
        itr = iter(self.predictor.parameters())
        for i in range(self.lenp):
            param = next(itr)
            w.append(param.to(self.device))
            if first_instance:
                wdot.append(torch.rand((param.shape)).to(self.device))
        if velocity == True: return w,wdot
        else: return w
    
    def flattenParamVector(self):
        itr = iter(self.predictor.parameters())         
        w = next(itr).view(-1)
        for i in range(1,self.lenp):
            w = torch.cat((w,(next(itr).view(-1))))
        return w

    def makeFMatrix(self):
            '''
            Subroutine to create F matrix for parameter state dynamics
            For memory efficiency, F is stored as a torch.sparse.Tensor
            '''           
            n = len(self.flat_wdot)
            #indexes of eye(n,n)
            i1 = [[0,n]]
            for i in range(1,n):
                i1.append([i,n+i])
            i1= torch.LongTensor(i1)
            # indexes of -eye(n,n)
            i2 = [[n,0]]
            for i in range(1,n):
                i2.append([n+i,i])
            i2= torch.LongTensor(i2)         
            # indexes of -beta(n,n)
            i3 = [[n,n]]
            for i in range(1,n):
                i3.append([n+i,n+i])
            i3 = torch.LongTensor(i3) 
             
            i = torch.cat((i1,i2,i3))
            #value list
            v = torch.Tensor(np.concatenate((np.ones(n),-1*np.ones(n),-self.beta*np.ones(n))))
            
            F = torch.sparse.FloatTensor(i.t(),v,torch.Size([2*n,2*n]))
            del i1,i2,i3,i,v
            return F
          
    def gradient(self):
        itr = iter(self.predictor.parameters())
        dJddw = 2.*self.hparams[1]*self.flat_wdot
        dJ_reg = 2.*self.hparams[2]*self.flat_w
        dJ = list(map(add, [next(itr).grad for i in range(self.len)],dJ_reg))
        self.dJ,self.dJddw = dJ, dJddw
        
    def additionalTermsLoss(self):
        return torch.add(self.hparams[1]*torch.dot(self.flat_wdot,self.flat_wdot),\
        self.hparams[2]*torch.dot(self.flat_w,self.flat_w))
    
    def assignNewState(self, xi):
        self.flat_w = torch.Tensor(xi[:len(self.flat_w)])
        self.flat_wdot = torch.Tensor(xi[len(self.flat_w):2*len(self.flat_w)])
        
    def loadStateDict(self):    
        new_state_dict = self.makeStateDict()
        del self.predictor
        self.predictor = self.p_type(*self.p_args).to(self.device)
        self.predictor.load_state_dict(new_state_dict)
        del new_state_dict
        
    def makeStateDict(self):
        d = {}
        k = 0
        for i,key in enumerate(self.predictor.state_dict().keys()):       
                num_el = torch.numel(self.predictor.state_dict()[key])
                d[key] = self.flat_w[k:k+num_el].view(self.predictor.state_dict()[key].shape)
                k += num_el
        return d
    
    def flattenGradient(self):
        dJ = self.dJ[0].view(-1)
        for i in range(1, self.len):
            dJ = torch.cat((dJ, self.dJ[i].view(-1)))
        return dJ.to(self.device), self.dJddw.to(self.device)
    
    def assignFlatGradient(self):
        self.flat_dJ,self.flat_dJddw = self.flattenGradient()
    
    def getConcatGradient(self):
        return torch.cat((self.flat_dJ, self.flat_dJddw))
    
    def fixInputOutput(self,x,y):
        self.x = x
        self.y = y
        
    def setXi(self):
        self.xi = torch.cat((self.flat_w.to(self.device), self.flat_wdot.to(self.device)))
        
    def getParamShape(self):
        return self.shape
    
    def pred_accuracy(self,testloader):
        tot = 0
        count = 0
        for i,d in enumerate(testloader):
            x,y = d
            x,y = x.to(0),y.to(0)
            _, idx = torch.max(torch.exp(self.predictor.forward(x)), 1)  
            for i in range(len(idx)):
                if idx[i] == y[i]:
                    count += 1
                tot += 1
        return count/tot
    
    def initializeRecord(self):
        for i in range(len(self.flat_w)):
            self.pW.append([])
            self.pWdot.append([])
            
    def recordLoss(self,main_loss,additional_terms_loss,delta_t):
        if self.time % delta_t == 0:
            self.pLoss.append(main_loss+additional_terms_loss)
    
    def plotLoss(self):
        plt.plot(self.pLoss, color='red')
        plt.ylabel('Loss')
        plt.xlabel('time (delta_t units)')
        
    def recordParameters(self,pW,pWdot,delta_t):
        if self.time % delta_t == 0:
            for i in range(len(pW)):
                self.pW[i].append(pW[i].cpu().detach().numpy()) 
                self.pWdot[i].append(pWdot[i].cpu().detach().numpy()) 
        
    def plotParameters(self):
        for i in range(len(self.pW)):
            plt.plot(self.pW[i])
            
    def plotVelocities(self):
        for i in range(len(self.pW)):
            plt.plot(self.pWdot[i])
    
    def perturb(self):
        pass
    
    def forward(self,t,xi):        
        #update internal flat w vectors
        self.assignNewState(xi)
        
        #use updated flat w to create new updated predictor
        self.loadStateDict()
        
        
        #loss and gradient
        yhat = self.predictor.forward(self.x)
        loss = self.criterion(yhat,self.y)#self.additionalTermsLoss())
        loss.backward()
        
        #optional saving for plots   
        self.recordLoss(loss,self.additionalTermsLoss(),self.time_delta)
        self.recordParameters(self.flat_w,self.flat_wdot,self.time_delta)
        del loss, yhat
        
        #manual gradient w.r.t wdot and assignment to flat gradient vectors
        self.gradient()
        self.assignFlatGradient()
        grad_flat = self.getConcatGradient()
        
        #actual ODE calcs
        dxdt = torch.zeros(len(grad_flat)).to(self.device)
        n = len(dxdt)//2
        dxdt[:n] = grad_flat[n:2*n]
        dxdt[n:2*n] = -1*grad_flat[:n] -1*self.beta*grad_flat[n:2*n]
        del grad_flat
        
        #timer for plotting purposes
        self.time += 1
        if self.time % 1000 == 0: print("odeint iter: {} ".format(self.time))
        dxdt = dxdt.detach().cpu().numpy()
        return dxdt
    
    def fit(self, trainloader, epoch=3, time_delta=1, iter_accuracy=10, ode_t=0.25, ode_step=10, criterion='nll'):
        """
        :trainloader: DataLoader with training data
        :epoch: number of training epochs
        :time_delta: time steps required for a single recording of loss and parameters. Higher is better for speed. If None, no plotting
        :method: can be either 'odeint' or 'adjoint'
        :iter_accuracy: iterations until test accuracy is displayed
        :ode_t: number of odeint time steps (per batch)
        :ode_step: number of odeint time steps (per batch)
        :criterion: nll for negative log-likelihood, mse for mean-square error
        """
        if criterion == 'nll': self.criterion = F.nll_loss 
        else: self.criterion = F.mse_loss         
        t = np.linspace(0., ode_t, ode_step) 
        if time_delta: self.time_delta = time_delta
        else: self.time_delta = float('inf')
        self.setXi()  
        
        for e in range(epoch): 
            for i, data in enumerate(trainloader):               
                x,y = data
                x,y = x.to(self.device),y.to(self.device)
                self.fixInputOutput(x,y)              
                func = self
                xi = solve_ivp(func, t, self.xi.cpu().detach().numpy())
                xi = [el[-1] for el in xi.y]
                self.xi = torch.Tensor(xi)
                self.assignNewState(xi)
                self.loadStateDict()
                self.count += 1                
                if self.count % iter_accuracy == 0 and self.count != 0:
                    print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                    print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))
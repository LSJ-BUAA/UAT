# coding=utf8
import torch
import torch.nn as nn
import math


class layer_1D_2DJ(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(layer_1D_2DJ, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0/output_dim/input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P6 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P7 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P8 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

        self.P9 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, aggr_func=torch.mean):
        # A = A.contiguous()

        BATCH_SIZE,_,L,K,_ = A.shape #BS,cin,L,K,K

        dev = A.device

        D = torch.eye(K).to(dev)
        ND = torch.ones([K,K]).to(dev)-D

        A1 = torch.matmul(self.P1, (D*A).view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, L, K, K)
        A2 = D * torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, L, K, 1)
        A3 = D * torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, L, 1, K)

        A4 = torch.matmul(self.P4, (ND*A).view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, L, K, K)
        A5 = ND * torch.matmul(self.P5, torch.diagonal(A, 0, -2, -1).reshape([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, L, K, 1)
        A6 = ND * torch.matmul(self.P6, torch.diagonal(A, 0, -2, -1).reshape([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, L, 1, K)
        A7 = ND * torch.matmul(self.P7, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, L, K, 1)
        A8 = ND * torch.matmul(self.P8, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, L, 1, K)

        A9 = D * torch.matmul(self.P9, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, K, K)

        A = A1 + 0.1*A2 + 0.1*A3 + A4 + 0.1*A5 + 0.1*A6 + 0.1*A7 + 0.1*A8 + 0.1*A9

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class y_net_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(y_net_GNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_1D_2DJ(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(layer_1D_2DJ(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, G, SNR, Pmax):
        
        BATCH_SIZE,L,K,_ = G.shape #BS,L,K,K

        G = G*SNR
        A = torch.stack([G.real, G.imag, Pmax.expand(BATCH_SIZE, L, K, K)], dim=1)#BS,3,L,K,K

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # BS,1,L,K,K

        P = A.mean(-2).squeeze(1)  # BS,L,K

        P = nn.Softplus()(P)

        row_sum = P.sum(-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(row_sum), Pmax.view(BATCH_SIZE,1,1) / (row_sum + 1e-12))
        P = P * scale # project to satisfy power constraint

        return P


class lambda_net_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(lambda_net_GNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_1D_2DJ(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(layer_1D_2DJ(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, G, SNR, Pmax):

        BATCH_SIZE,L,K,_ = G.shape #BS,L,K,K

        G = G*SNR
        A = torch.stack([G.real, G.imag, Pmax.expand(BATCH_SIZE, L, K, K)], dim=1)#BS,3,L,K,K

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # BS,1,L,K,K

        lam = A.mean([-2,-3]).squeeze(1)  # BS,K
        lam = nn.Softplus()(lam)
        return lam


class layer_FNN(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(layer_FNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer


        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self, A):
        A = self.lin(A)

        if self.is_BN:
            A = self.batch_norms(A)
        if self.is_transfer:
            A = self.activation(A)
        return A


class y_net_FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(y_net_FNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_FNN(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(layer_FNN(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, G, SNR, Pmax):

        BATCH_SIZE,L,K,_ = G.shape #BS,L,K,K

        G = G*SNR
        A = torch.stack([G.real, G.imag], dim=1)#BS,2,L,K,K

        A = A.view(BATCH_SIZE,-1)
        A = torch.cat([A, Pmax.view(BATCH_SIZE,1)],1)#BS, 2LK^2+1

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # BS, LK

        P = A.view(BATCH_SIZE,L,K)

        P = nn.Softplus()(P)

        row_sum = P.sum(-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(row_sum), Pmax.view(BATCH_SIZE,1,1) / (row_sum + 1e-12))
        P = P * scale

        return P


class lambda_net_FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(lambda_net_FNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_FNN(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(layer_FNN(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, G, SNR, Pmax):

        BATCH_SIZE, L, K, _ = G.shape  # BS,L,K,K

        G = G * SNR
        A = torch.stack([G.real, G.imag], dim=1)  # BS,2,L,K,K

        A = A.view(BATCH_SIZE, -1)
        A = torch.cat([A, Pmax.view(BATCH_SIZE, 1)],1)  # BS, 2LK^2+1

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # BS,K

        lam = A  # BS,K

        lam = nn.Softplus()(lam)
        return lam


class layer_CNN(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(layer_CNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        self.CC = nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding='same',
                bias=True,
                padding_mode='circular'#pad=2
                # padding_mode='zeros'#pad=1
            )

    def forward(self, A, aggr_func=torch.mean):
        A = self.CC(A)

        if self.is_BN:
            A = self.batch_norms(A)
        if self.is_transfer:
            A = self.activation(A)
        return A



class y_net_ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(y_net_ResNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]

        # Ref. "Joint User Association and Power Control for Cell-Free Massive MIMO"
        self.layers.append(layer_CNN(self.dim[0], self.dim[1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))  # 1
        self.layers.append(layer_CNN(self.dim[1], self.dim[2], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 2
        self.layers.append(layer_CNN(self.dim[2], self.dim[3], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 3
        self.layers.append(layer_CNN(self.dim[3], self.dim[4], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 4
        self.layers.append(layer_CNN(self.dim[4], self.dim[5], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 5
        self.layers.append(layer_CNN(self.dim[5], self.dim[6], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 6
        self.layers.append(layer_CNN(self.dim[6], self.dim[7], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 7
        self.layers.append(layer_CNN(self.dim[7], self.dim[8], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))  # 8


    def forward(self, G, SNR, Pmax):

        BATCH_SIZE,L,K,_ = G.shape #BS,L,K,K

        G = G*SNR
        A = torch.stack([G.real, G.imag, Pmax.expand(BATCH_SIZE, L, K, K)], dim=1)#BS,3,L,K,K


        A0 = A.view(BATCH_SIZE,3*L,K,K) 

        A1 = self.layers[0](A0)
        A2 = self.layers[1](A1)
        A3 = self.layers[2](A2) + A1 # Res

        A4 = self.layers[3](A3)
        A5 = self.layers[4](A4) + A3 # Res

        A6 = self.layers[5](A5)
        A7 = self.layers[6](A6) + A5 # Res
        A8 = self.layers[7](A7) 

        P = A8.diagonal(dim1=-2, dim2=-1) 

        P = nn.Softplus()(P)

        row_sum = P.sum(-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(row_sum), Pmax.view(BATCH_SIZE,1,1) / (row_sum + 1e-12))
        P = P * scale

        return P


class lambda_net_ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(lambda_net_ResNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]

        self.layers.append(layer_CNN(self.dim[0], self.dim[1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))  # 1
        self.layers.append(layer_CNN(self.dim[1], self.dim[2], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 2
        self.layers.append(layer_CNN(self.dim[2], self.dim[3], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))  # 3
        self.layers.append(layer_CNN(self.dim[3], self.dim[4], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))  # 4

    def forward(self, G, SNR, Pmax):

        BATCH_SIZE,L,K,_ = G.shape #BS,L,K,K

        G = G*SNR
        A = torch.stack([G.real, G.imag, Pmax.expand(BATCH_SIZE, L, K, K)], dim=1)#BS,3,L,K,K

        A0 = A.view(BATCH_SIZE, 3*L , K, K)  
        A1 = self.layers[0](A0)
        A2 = self.layers[1](A1)
        A3 = self.layers[2](A2) + A1 # Res
        A4 = self.layers[3](A3)

        lam = A4.diagonal(dim1=-2, dim2=-1).squeeze(1)  # BS,K

        lam = nn.Softplus()(lam)
        return lam


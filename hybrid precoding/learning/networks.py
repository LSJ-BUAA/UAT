import torch
import torch.nn as nn
import math
from utils import *


class layer_agnn(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True, is_attention=True):
        super(layer_agnn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer
        self.is_attention = is_attention

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0/output_dim/input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        # self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        # self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

        if is_attention:
            self.Q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini) #P6
            self.K = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini) #P7

    def forward(self, A, aggr_func=torch.mean):
        BATCH_SIZE,_,M,K,NR,NT = A.shape

        A = A.contiguous()

        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, M, K, NR, NT)
        # A2 = torch.matmul(self.P2, aggr_func(A, 2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, K, NR, NT)
        A3 = torch.matmul(self.P3, aggr_func(A, 5).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, K, NR, 1)
        # A4 = torch.matmul(self.P4, aggr_func(A, 4).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, K, 1, NT)

        if self.is_attention:
            MeanA4 = torch.mean(A, 4)
            q = torch.matmul(self.Q, MeanA4.view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, M, K, NT)
            k = torch.matmul(self.K, MeanA4.view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, M, K, NT)
            alpha = nn.Tanh()(torch.matmul(q, k.transpose(-1,-2))/NT)

            v = torch.matmul(self.P5, MeanA4.view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, K,  NT)
            A5 = torch.matmul(alpha, v).view([BATCH_SIZE, self.output_dim, M, K, 1, NT]) / K

            A = A1 + 0.1 * A3 + A5

        else:
            A5 = torch.matmul(self.P5, aggr_func(A, [3,4]).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, 1, 1, NT)
            A = A1 + A3 + A5

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A.view(BATCH_SIZE, self.output_dim, -1)).view(BATCH_SIZE, self.output_dim, M,K,NR,NT)
        return A


class AGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_attention, NRF):
        super(AGNN, self).__init__()
        self.NRF = NRF
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_agnn(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True, is_attention=is_attention))
            else:
                self.layers.append(layer_agnn(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False, is_attention=is_attention))

    def forward(self, A, sqrtSNR, equal_user):

        A = A*sqrtSNR
        BATCH_SIZE, K, NT = A.shape

        A = torch.stack([A.real, A.imag], dim=1)

        A = A.view(BATCH_SIZE, 2, 1, K, 1, NT)

        NRF = self.NRF

        # update layers
        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # A(BS,2+4NRF,M,K,NR,NT)

        # output layer: y for W_BB, z for W_RF
        # should output WRF(BS,2,1,1,NT,NRF), WBB(BS,2,M,K,NRF,1)

        # WBB
        y1 = torch.mean(A[:, 0:NRF], dim=[4, 5], keepdim=True).transpose(1, 4)
        y2 = torch.mean(A[:, NRF:2 * NRF], dim=[4, 5], keepdim=True).transpose(1, 4)

        y = torch.cat([y1, y2], dim=1)

        # to satisfy WRF constraint
        z1 = torch.mean(A[:, 2 * NRF:3 * NRF], dim=[2, 3, 4]).transpose(-1, -2).view([BATCH_SIZE, 1, 1, 1, NT, NRF])
        z2 = torch.mean(A[:, 3 * NRF:4 * NRF], dim=[2, 3, 4]).transpose(-1, -2).view([BATCH_SIZE, 1, 1, 1, NT, NRF])

        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo
        z = torch.cat([z1, z2], dim=1)

        # to satisfy power constraints
        W = complex_matmul(z, y) #W(BS,2,M,K,NT,1)

        # allocate equal power to each user
        # during the first several epochs to avoid local minima
        if equal_user:
            temp = math.sqrt(K) * torch.sqrt(torch.sum(ampli2(W), dim=[1, 3])).view(BATCH_SIZE, 1, 1, K, 1, 1)
        else:
            temp = torch.sqrt(torch.sum(ampli2(W), dim=[1, 2, 3])).view(BATCH_SIZE, 1, 1, 1, 1, 1)

        y = y / temp

        return y, z


class Layer_3DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,permutation_size1,permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,permutation_size1,1, permutation_size3)
        A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,1,permutation_size2, permutation_size3)

        A = A1 + 0.1*A2 + 0.1*A3 + 0.1*A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, NRF):
        super(GNN3D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.Ns = NRF

        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False,is_transfer=False))

    def forward(self, A, sqrtSNR, equal_user):

        A = A * sqrtSNR
        BATCH_SIZE, K, Nt = A.shape

        A = torch.stack([A.real, A.imag], dim=1)

        Ns = self.Ns

        vfeature = torch.arange(-2, 2, 4 / Ns).view(1, 1, 1, 1, Ns).to(A.device) # input layer: the Virtual Feature

        A = A.view(BATCH_SIZE,2,K,Nt,1)+vfeature # input layer

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A, permutation_size1=K, permutation_size2=Nt,permutation_size3=Ns, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0], dim=1)
        z2 = torch.mean(A[:, 1], dim=1)
        y1 = torch.mean(A[:, 2], dim=2)
        y2 = torch.mean(A[:, 3], dim=2)

        # to satisfy constraints
        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y1 = y1.view([BATCH_SIZE, 1, K, Ns])
        y2 = y2.view([BATCH_SIZE, 1, K, Ns])
        z1 = z1.view([BATCH_SIZE, 1, Nt, Ns])
        z2 = z2.view([BATCH_SIZE, 1, Nt, Ns])
        y = torch.cat([y1, y2], dim=1)
        z = torch.cat([z1, z2], dim=1)

        y = y.view(BATCH_SIZE,2,1,K,Ns,1)
        z = z.view(BATCH_SIZE,2,1,1,Nt,Ns)


        # to satisfy power constraints
        W = complex_matmul(z, y)  # W(BS,2,M,K,NT,1)

        # allocate equal power to each RB or user
        # during the first several epochs to avoid local minima
        if equal_user:
            temp = math.sqrt(K) * torch.sqrt(torch.sum(ampli2(W), dim=[1, 3])).view(BATCH_SIZE, 1, 1, K, 1, 1)
        else:
            temp = torch.sqrt(torch.sum(ampli2(W), dim=[1, 2, 3])).view(BATCH_SIZE, 1, 1, 1, 1, 1)

        y = y / temp
        return y, z


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, NRF):
        super(CNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.Ns = NRF

        dim = [input_dim] + list(hidden_dim)
        for i in range(len(dim) - 1):
            self.convs.append(nn.Conv2d(
                in_channels=dim[i],
                out_channels=dim[i + 1],
                kernel_size=2,
                padding='same',
                bias=True,
                padding_mode='circular'
                # padding_mode='zeros'
            ))
            self.batch_norms.append(nn.BatchNorm2d(dim[i + 1]))

        D = 512
        self.linear_layer = nn.Linear(hidden_dim[-1]*4*16, D)
        self.linear_BN = nn.BatchNorm1d(D)
        self.output_linear = nn.Linear(D, output_dim)

        self.activation = nn.ReLU()

    def forward(self, A, sqrtSNR, equal_user):

        A = A * sqrtSNR
        BATCH_SIZE, K, Nt = A.shape

        A = torch.stack([A.real, A.imag], dim=1)

        x = A.view(BATCH_SIZE,2,K,Nt)

        for layer in range(len(self.convs)):
            x = self.convs[layer](x)
            x = self.batch_norms[layer](x)
            x = self.activation(x)

        x = x.reshape(BATCH_SIZE,-1)#FC
        x = self.linear_layer(x)
        x = self.linear_BN(x)
        x = self.activation(x)

        output = self.output_linear(x)#FC out
        Ns = self.Ns

        # outputs
        y1 = output[:, 0: K * Ns]
        y2 = output[:, K * Ns: 2 * K * Ns]
        y = torch.stack([y1,y2],1).view(BATCH_SIZE,2,1,K,Ns,1)

        z1 = output[:, 2 * K * Ns:2 * K * Ns + Ns * Nt]
        z2 = output[:, 2 * K * Ns + Ns * Nt: 2 * K * Ns + 2 * Ns * Nt]

        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo
        z = torch.stack([z1, z2], 1).view(BATCH_SIZE,2,1,1,Nt,Ns)

        # to satisfy power constraints
        W = complex_matmul(z, y)  # W(BS,2,M,K,NT,1)

        # allocate equal power to each RB or user
        # during the first several epochs to avoid local minima
        if equal_user:
            temp = math.sqrt(K) * torch.sqrt(torch.sum(ampli2(W), dim=[1, 3])).view(BATCH_SIZE, 1, 1, K, 1, 1)
        else:
            temp = torch.sqrt(torch.sum(ampli2(W), dim=[1, 2, 3])).view(BATCH_SIZE, 1, 1, 1, 1, 1)

        y = y / temp
        return y, z


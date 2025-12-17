# coding=utf8
import torch
import torch.nn as nn
import math
import h5py
import numpy as np


def cal_beamforming(H, BF='MRT', Pmax=1, sigma2=1):
    BS, L, NT, K = H.shape
    if BF == 'MRT':
        W = H
    elif BF == 'ZF':
        if NT < K:
            W = torch.matmul(torch.linalg.inv(torch.matmul(H,H.mH)),H)
        else:
            W = torch.matmul(H,torch.linalg.inv(torch.matmul(H.mH,H)))
    elif BF == 'RZF':
        if NT < K:
            W = torch.matmul(torch.linalg.inv(torch.matmul(H,H.mH) + K*sigma2/Pmax*torch.eye(NT)),H)
        else:
            W = torch.matmul(H,torch.linalg.inv(torch.matmul(H.mH,H) + K*sigma2/Pmax*torch.eye(K)))
    else:
        print('Beamforming error')

    W = W/torch.sqrt(torch.sum(torch.abs(W)**2,2,keepdim=True)) #power norm, [BS,L,NT,K]
    G = torch.matmul(H.mH,W)
    return G


def calEE(G, P, sigma2, Pfix, a, b, B, SEqos, device, training=False):
    # equivalent channel G(BS,L,K,I), power allocation P(BS,L,I), where I = K
    BS, L, K, _ = G.shape
    Psum = torch.sum(P, [1,2]) #BS

    C = torch.sqrt(P+1e-12) #BS,L,I

    C = C.unsqueeze(-1) #BS,L,I,1
    C = C.transpose(1,2)#BS,I,L,1

    G = G.permute(0,3,2,1) #BS,I,K,L

    C = C.to(torch.complex64)
    GC = torch.matmul(G,C) #BS,I,K,1
    GC = GC.squeeze(3)  #BS,I,K

    GC2 = torch.abs(GC)**2 #BS,I,K
    D = torch.eye(K).to(device)*GC2 #BS,I,K
    sumD = torch.sum(D,1)#BS,K
    sumQ = torch.sum(GC2,1)#BS,K
    SINR = sumD/(sumQ-sumD+sigma2)#BS,K
    SEperUE = torch.log2(1.0 + SINR)#BS,K

    SE = torch.sum(SEperUE,1) #BS
    Ptotal = Pfix + a*Psum + b*B*SE #BS
    EE = B*SE/Ptotal #BS

    if training:
        return torch.mean(EE), EE, SE, Psum, SEperUE, torch.tensor(0)

    else:
        SEqos = SEqos - 1e-3
        mask = SEperUE < SEqos #check SE QoS satisfying
        any_smaller = mask.any(dim=1)
        EE[any_smaller] = 0

        Violation_ratio = any_smaller.float().mean() * 100

        return torch.mean(EE), EE, SE, Psum, SEperUE, Violation_ratio


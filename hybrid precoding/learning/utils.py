import torch
import torch.nn as nn
import math


def complex_matmul(A,B):
    # Multiplication of two complex matrices
    b1, _, _, k1, h1, _ = A.size() # BS, 2, M or 1, K or 1, h, p
    _, _, _, k2, _, w2 = B.size() # BS, 2, M or 1, K or 1, p, w
    return torch.stack([torch.matmul(A[:, 0, :, :, :, :],B[:, 0, :, :, :, :])-torch.matmul(A[:, 1, :, :, :, :],B[:, 1, :, :, :, :]),
                      torch.matmul(A[:, 0, :, :, :, :],B[:, 1, :, :, :, :])+torch.matmul(A[:, 1, :, :, :, :],B[:, 0, :, :, :, :])],dim=1).view(b1,2,1,max(k1,k2),h1,w2)


def ampli2(A):
    # Squared Frobenius norm (squeeze one dimension)
    return A[:,0,:,:,:,:]**2+A[:,1,:,:,:,:]**2


def cal_sumrate(H, WBB, WRF, sigma2, device):
    # Calculate the averaged spectral efficiency over a batch
    # SIZE: H(BS,2,M,K,NR,NT), WRF(BS,2,1,1,NT,NRF), WBB(BS,2,M,K,NRF,1)
    # M (number of subcarriers) and NR (number of receive antennas) must equal 1 in this work

    BATCH_SIZE, K, NT = H.shape
    H = torch.stack([H.real, H.imag], dim=1)
    H = H.view(BATCH_SIZE, 2, 1, K, 1, NT)

    Heff = complex_matmul(H,WRF)#BS,2,M,K,1,NRF

    Heff = Heff.transpose(-2,-3)
    WBB = WBB.transpose(-1,-3)

    Q = complex_matmul(Heff, WBB)
    Q2 = ampli2(Q)

    D = torch.eye(K).to(device) * Q2
    sumD = torch.sum(D, dim=-1, keepdim=True)
    sumQ = torch.sum(Q2, dim=-1, keepdim=True)
    sinr = sumD/(sumQ-sumD+sigma2)
    rate = torch.sum(torch.log2(1.0 + sinr))

    return rate/BATCH_SIZE
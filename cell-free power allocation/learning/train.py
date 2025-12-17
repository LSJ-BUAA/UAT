import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys
import os
import math
from utils import calEE, cal_beamforming
from network import y_net_GNN, lambda_net_GNN, y_net_ResNet, lambda_net_ResNet, y_net_FNN, lambda_net_FNN
import h5py
import random

# train a DNN using PDL or using EUAT with a loaded model


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    time_start = time.time()

    K = 10 # number of single-antenna users
    L = 25 # number of APs
    NT = 4 # number of antennas at each AP

    D = 500 # side length for train
    PmaxdBm = 20+35.3*math.log10(D/500) # in dBm
    Pmax = 10**((PmaxdBm-30)/10) # in Watt

    sigma2 = 1  # don't change

    Pfix = L * (NT * 0.2 + 0.825)
    a = 1 / 0.4
    b = 0.25e-9 * 1e6
    B = 20e6 / 1e6  # in MHz

    SEqos = 1  # QoS Constraint for each user, in bps/Hz

    train_number = 20000
    test_number = min(train_number, 2000)

    # Load datasets
    set_num1 = 20000
    train_H = []
    train_filenames = ['../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(D)+'_num'+str(set_num1)+'.mat']
    for filename in train_filenames:
        with h5py.File(filename, 'r') as f:
            real_part = np.array(f['Hset']['real'])
            imag_part = np.array(f['Hset']['imag'])
            H = real_part + 1j * imag_part
            H = np.swapaxes(H, -1, -2)
            train_H.append(torch.from_numpy(H[0:train_number]))


    PmaxTestSet = [10**((20+35.3*math.log10(100/500)-30)/10),
                   10**((20+35.3*math.log10(200/500)-30)/10),
                   10**((20+35.3*math.log10(400/500)-30)/10),
                   10**((20+35.3*math.log10(500/500)-30)/10),
                   10**((20+35.3*math.log10(800/500)-30)/10),
                   10**((20+35.3*math.log10(1000/500)-30)/10)]
    set_num2 = 2000
    test_H = []
    test_filenames = ['../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(100)+'_num'+str(set_num2)+'.mat',
                      '../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(200)+'_num'+str(set_num2)+'.mat',
                      '../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(400)+'_num'+str(set_num2)+'.mat',
                      '../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(500)+'_num'+str(set_num2)+'.mat',
                      '../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(800)+'_num'+str(set_num2)+'.mat',
                      '../channels/datasets/CHleng_NT'+str(NT)+'_K'+str(K)+'_L'+str(L)+'_D'+str(1000)+'_num'+str(set_num2)+'.mat']

    for filename in test_filenames:
        with h5py.File(filename, 'r') as f:
            real_part = np.array(f['Hset']['real'])
            imag_part = np.array(f['Hset']['imag'])
            H = real_part + 1j * imag_part
            H = np.swapaxes(H, -1, -2)
            test_H.append(torch.from_numpy(H[0:test_number]))

    BF = 'RZF' # beamforming method
    train_G = []
    for m in range(len(train_filenames)):
        train_G.append(cal_beamforming(train_H[m], BF))  # train_number, L, K, K

    test_G = []
    for m in range(len(test_filenames)):
        test_G.append(cal_beamforming(test_H[m], BF))  # test_number, L, K, K

    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    MAX_EPOCH = 300
    BS = min(100, train_number)
    LEARNING_RATE = 1e-3

    A = 1  # dynamic Pmax range (+-10*A dB) for training, adjust according to the required generalization range

    modelName = 'GNN'
    # modelName = 'ResNet'
    # modelName = 'FNN'

    if modelName == 'GNN':
        y_layer = [64]*4
        y_net = y_net_GNN(input_dim=3, hidden_dim=y_layer, output_dim=1)
    elif modelName == 'ResNet':
        y_layer = [64]*7
        y_net = y_net_ResNet(input_dim=3*L, hidden_dim=y_layer, output_dim=1*L)
    elif modelName == 'FNN':
        y_layer = [256]*6
        y_net = y_net_FNN(input_dim=2*L*K*K+1, hidden_dim=y_layer, output_dim=L*K)
    else:
        pass

    y_net.to(device)
    y_optimizer = torch.optim.Adam([{'params': y_net.parameters()}], lr=LEARNING_RATE, weight_decay=0)
    y_scheduler = torch.optim.lr_scheduler.MultiStepLR(y_optimizer, [30, 60, 90, 120], gamma=0.3, last_epoch=-1)


    timestr = time.strftime("%m%d%H%M%S", time.localtime())
    print('model', modelName, 'BF', BF, 'K', K, 'NT', NT, 'L', L, 'D',D, 'Pmax', Pmax, 'range', A, 'sigma2', sigma2, 'Pfix', Pfix, 'a', a, 'b', b, 'B', B, 'SEqos', SEqos, 'time', timestr)
    print('train_number', train_number, 'test_number', test_number, 'batch_size', BS, 'ini_learning_rate', LEARNING_RATE, 'y_layer', y_layer)

    print(train_filenames)
    print(test_filenames)

    is_PDL = True # True if QoS constraint exists
    if is_PDL:
        PDL_LEARNING_RATE = LEARNING_RATE * .1

        if modelName == 'GNN':
            lam_layer = [32] * 3
            lam_net = lambda_net_GNN(input_dim=3, hidden_dim=lam_layer, output_dim=1)
        elif modelName == 'ResNet':
            lam_layer = [64] * 3
            lam_net = lambda_net_ResNet(input_dim=3*L, hidden_dim=lam_layer, output_dim=1)
        elif modelName == 'FNN':
            lam_layer = [128] * 4
            lam_net = lambda_net_FNN(input_dim=2*L*K*K+1, hidden_dim=lam_layer, output_dim=K)
        else:
            pass

        lam_net.to(device)
        lam_optimizer = torch.optim.Adam([{'params': lam_net.parameters()}], lr=PDL_LEARNING_RATE, weight_decay=0)
        lam_scheduler = torch.optim.lr_scheduler.MultiStepLR(lam_optimizer, [60,90,120], gamma=0.3, last_epoch=-1)

        amp = 50 # or 20
        # emphasize the samples that do not satisfy constraints, to accelerate convergence

        print('PDL', 'lam_layer', lam_layer, 'LR',PDL_LEARNING_RATE, 'amp', amp)


    # hyper-parameters for EUAT
    epoch_UAT = 9999
    T_UAT = 5
    alpha_UAT = 0.1 # or 0.2
    is_deH = False  # always False
    is_minusG = True  # always True
    is_plusG = True # always True
    w_plusG = 1 # balancing coefficient w
    k_plusG = 10 # approximation degree m

    is_EUAT = False # Flase for PDL, True for EUAT

    if is_EUAT:
        address = 'model40_' + '1129185351' + 'RZF' + '_K' + str(K) + '_NT' + str(NT) + '_L' +str(L) + '_D' +str(D) + '.ckp'
        print('EUAT', 'T', T_UAT, 'alpha', alpha_UAT, 'w', w_plusG, 'm', k_plusG)
        print(address)
        y_net.load_state_dict(torch.load(address)['y_net'])
        # y_optimizer.load_state_dict(torch.load(address)['y_optimizer'])
        # y_scheduler.load_state_dict(torch.load(address)['y_scheduler'])
        lam_net.load_state_dict(torch.load(address)['lam_net'])
        # lam_optimizer.load_state_dict(torch.load(address)['lam_optimizer'])
        # lam_scheduler.load_state_dict(torch.load(address)['lam_scheduler'])
        epoch_UAT = 0


    sys.stdout.flush()
    eta = SEqos * torch.ones(BS, K).to(device)
    highestEE = 0

    for epoch in range(MAX_EPOCH):

        index = [i for i in range(train_number)]  # shuffle training set
        shuffle(index)
        for m in range(len(train_filenames)):
            train_G[m] = train_G[m][index]

        y_net.train()
        if is_PDL:
            lam_net.train()

        for bb in range(int(train_number / BS)):
            for m in range(len(train_filenames)):
                batch_x = train_G[m][bb * BS:(bb + 1) * BS].to(device)#BS,L,K,K

                normG = torch.sum(torch.abs(batch_x)**2, dim=[1,2,3],keepdim=True)/sigma2 #BS,1,1,1

                ran = 10 ** torch.empty(BS,1,1,1, device=device).uniform_(-A, A)
                normG = normG * ran
                PmaxRand = Pmax/ran

                sqrtnorm = torch.sqrt(normG)
                batch_x = batch_x/sqrtnorm

                if epoch >= epoch_UAT:
                    feature = batch_x.clone()
                    feature.requires_grad = True

                y_pred = y_net(feature if epoch>=epoch_UAT else batch_x, sqrtnorm, PmaxRand)

                EE, _, _, _, SEperUE, _ = calEE(batch_x*sqrtnorm if (not is_deH) or epoch<epoch_UAT else feature*sqrtnorm, y_pred, 1, Pfix, a, b, B, eta, device, training=True)

                if is_PDL:
                    lam = lam_net(batch_x, sqrtnorm, PmaxRand)
                    y_net.zero_grad()
                    lam_net.zero_grad()

                    if epoch >= epoch_UAT:

                        temp = -EE
                        temp.backward(retain_graph=True)  #compute the grad of EE wrt x and theta

                        grad_x = feature.grad.clone()  #save the grad of EE wrt x
                        feature.grad.zero_() # and clear the grad on x

                        temp = torch.sum(lam * -1*nn.LeakyReLU(negative_slope=amp)(SEperUE-eta))/BS
                        temp.backward()
                        feature.grad.zero_()

                        for param in lam_net.parameters():
                            param.grad = -param.grad
                        y_optimizer.step()
                        lam_optimizer.step()

                    else:
                        loss = -EE + torch.sum(lam * -1*nn.LeakyReLU(negative_slope=amp)(SEperUE-eta))/BS
                        loss.backward()

                        for param in lam_net.parameters():
                            param.grad = -param.grad
                        y_optimizer.step()
                        lam_optimizer.step()

                else:
                    loss = -EE
                    y_net.zero_grad()
                    loss.backward()

                    if epoch >= epoch_UAT:
                        grad_x = feature.grad.clone()
                        feature.grad.zero_()

                    y_optimizer.step()


                # EUAT BEGIN #
                if epoch >= epoch_UAT:
                    for t in range(T_UAT):

                        if is_minusG:
                            y_pred_de = y_pred.detach()
                            _, _, _, _, SEperUE, _ = calEE(feature*sqrtnorm, y_pred_de, 1, Pfix, a, b, B, eta, device, training=True)  # get the partial derivatives of G wrt x
                            lam_de = lam.detach()
                            temp = torch.sum(lam_de * SEperUE)/BS
                            temp.backward()
                            grad_x = grad_x + feature.grad
                            feature.grad.zero_()

                        if is_plusG:
                            y_pred = y_net(feature, sqrtnorm, PmaxRand)
                            _, _, _, _, SEperUE, _ = calEE(feature*sqrtnorm, y_pred, 1, Pfix, a, b, B, eta, device, training=True)  # get the partial derivatives of G wrt x
                            temp = -1 * w_plusG * torch.sum( nn.Sigmoid()(k_plusG*(eta-SEperUE.detach())) * SEperUE)/BS
                            temp.backward()
                            grad_x = grad_x + feature.grad
                            feature.grad.zero_()


                        norm = torch.sqrt(torch.sum(torch.abs(grad_x)**2, dim=[1,2,3], keepdim=True))  #update x
                        feature = feature + alpha_UAT * grad_x/norm #* torch.sqrt(torch.sum(torch.abs(feature)**2, dim=[1,2,3], keepdim=True))
                        feature = feature/torch.sqrt(torch.sum(torch.abs(feature)**2, dim=[1,2,3], keepdim=True))

                        #PDL
                        feature = feature.clone().detach().requires_grad_(True)
                        batch_x = feature.clone().detach()

                        y_pred = y_net(feature, sqrtnorm, PmaxRand)

                        EE, _, SE, Psum, SEperUE, VR = calEE(batch_x*sqrtnorm if (not is_deH) else feature*sqrtnorm, y_pred, 1, Pfix, a, b, B, eta, device, training=True)

                        if is_PDL:
                            lam = lam_net(batch_x, sqrtnorm, PmaxRand)

                            y_net.zero_grad()
                            lam_net.zero_grad()

                            temp = -EE
                            temp.backward(retain_graph=True)

                            grad_x = feature.grad.clone()
                            feature.grad.zero_()

                            temp = torch.sum(lam * -1 * nn.LeakyReLU(negative_slope=amp)(SEperUE - eta)) / BS
                            temp.backward()

                            for param in lam_net.parameters():
                                param.grad = -param.grad

                            y_optimizer.step()
                            lam_optimizer.step()

                        else:
                            loss = -EE
                            y_net.zero_grad()
                            loss.backward()

                            grad_x = feature.grad.clone()
                            feature.grad.zero_()

                            y_optimizer.step()
                #EUAT END #

        y_scheduler.step()
        if is_PDL:
            lam_scheduler.step()

        if epoch % 1 == 0: 
            y_net.eval()

            with torch.no_grad():
                sum_train_loss = [0]*len(train_filenames)
                sum_train_error = [0]*len(train_filenames)

                sum_test_loss = [0]*len(test_filenames)
                sum_test_error = [0]*len(test_filenames)

                for bb in range(int(test_number / BS)):
                    for m in range(len(train_filenames)):
                        batch_x = train_G[m][bb * BS:(bb + 1) * BS].to(device)

                        normG = torch.sum(torch.abs(batch_x) ** 2, dim=[1, 2, 3], keepdim=True) / sigma2  # BS,1,1,1
                        sqrtnorm = torch.sqrt(normG)
                        batch_x = batch_x / sqrtnorm

                        y_pred = y_net(batch_x, sqrtnorm, Pmax*torch.ones_like(normG))
                        EE, _, SE, Psum, SEperUE, VR = calEE(batch_x*sqrtnorm, y_pred, 1, Pfix, a, b, B, eta, device, training=False)
                        sum_train_loss[m] = sum_train_loss[m] + EE.detach().cpu().numpy()
                        sum_train_error[m] = sum_train_error[m] + VR.detach().cpu().numpy()

                    for m in range(len(test_filenames)):
                        batch_x = test_G[m][bb * BS:(bb + 1) * BS].to(device)

                        normG = torch.sum(torch.abs(batch_x) ** 2, dim=[1, 2, 3], keepdim=True) / sigma2  # BS,1,1,1
                        sqrtnorm = torch.sqrt(normG)
                        batch_x = batch_x / sqrtnorm

                        y_pred = y_net(batch_x, sqrtnorm, PmaxTestSet[m]*torch.ones_like(normG))
                        EE, _, _, _, _, VR = calEE(batch_x*sqrtnorm, y_pred, 1, Pfix, a, b, B, eta, device, training=False)
                        sum_test_loss[m] = sum_test_loss[m] + EE.detach().cpu().numpy()
                        sum_test_error[m] = sum_test_error[m] + VR.detach().cpu().numpy()

                time_end = time.time()
                print(epoch, "{:.2f}".format(time_end - time_start), end=' ')
                for m in range(len(train_filenames)):
                    print(f"{sum_train_loss[m] / (test_number / BS):.4f}", end=' ')
                    print(f"{sum_train_error[m] / (test_number / BS):.2f}", end=' ')
                print('|', end=' ')
                for m in range(len(test_filenames)):
                    print(f"{sum_test_loss[m] / (test_number / BS):.4f}", end=' ')
                    print(f"{sum_test_error[m] / (test_number / BS):.2f}", end=' ')
                print()
                sys.stdout.flush()
                time_start = time.time()

            
            if epoch >= 20 and sum_test_loss[3]>highestEE:
                highestEE = sum_test_loss[3]
                if epoch <= 40: # save the best model trained by PDL, before E_A=40 epoch, to further train it with EUAT
                    torch.save({'y_net': y_net.state_dict(), 'y_optimizer': y_optimizer.state_dict(), 'y_scheduler': y_scheduler.state_dict(),
                                'lam_net': lam_net.state_dict(),'lam_optimizer': lam_optimizer.state_dict(), 'lam_scheduler': lam_scheduler.state_dict()},
                               'model40_' + timestr + BF + '_K' + str(K) + '_NT' + str(NT) + '_L' +str(L) + '_D' +str(D) + '.ckp')
                else: # save the best model to evaluate the performance of PDL or UAT
                    torch.save({'y_net': y_net.state_dict(), 'y_optimizer': y_optimizer.state_dict(), 'y_scheduler': y_scheduler.state_dict(),
                                'lam_net': lam_net.state_dict(), 'lam_optimizer': lam_optimizer.state_dict(), 'lam_scheduler': lam_scheduler.state_dict()},
                               'model_' + timestr + BF + '_K' + str(K) + '_NT' + str(NT) + '_L' + str(L) + '_D' + str(D) + '.ckp')


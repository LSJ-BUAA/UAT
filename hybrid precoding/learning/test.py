import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys
import math
import h5py
from utils import cal_sumrate
from networks import AGNN, GNN3D, CNN

# Test a well-trained DNN


if __name__ == '__main__':
    time_start = time.time()
    print('Test')

    K = 4 #number of users
    NT = 16 #number of BS antennas

    NRF = 6 #number of BS RF chains

    Ptot = 25  # BS transmit power in [dBm]
    sigma2 = 1  # don't change, the noise power has been considered in channel coefficient

    train_number = 2000  # just as a placeholder, not actually training
    test_number = min(train_number, 2000)  # number of test samples


    ##load datasets
    set_num1 = 100000
    set_num2 = 2000

    fc_train = 3.5e9
    train_H = []
    train_filenames = ['../channels/datasets/CHfc_NT' + str(NT) + '_K' + str(K) + '_fc' + str(int(fc_train)) + '_num' + str(set_num1) + '.mat']
    for filename in train_filenames:
        with h5py.File(filename, 'r') as f:
            real_part = np.array(f['Hset']['real'])
            imag_part = np.array(f['Hset']['imag'])
            H = real_part + 1j * imag_part
            Hset = torch.from_numpy(H[0:train_number])
            Hset = Hset * 10**(Ptot/20) * fc_train/3.5e9
            if fc_train > 6e9:
                Hset = Hset * math.sqrt(5)
            train_H.append(Hset)
    
    test_H = []  # test with the channels of different antenna spacing, as an example
    test_filenames = ['../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.1) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.2) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.3) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.4) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.5) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(0.75) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(1) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(1.5) + '_num' + str(set_num2) + '.mat',
                      '../channels/datasets/CHspacing_NT' + str(NT) + '_K' + str(K) + '_d' + str(2) + '_num' + str(set_num2) + '.mat']
    # m = 0
    # testfcset = [0.7e9, 2.1e9, 2.6e9, 3.5e9, 4.9e9, 28e9*math.sqrt(5), 39e9*math.sqrt(5), 60e9*math.sqrt(5), 73e9*math.sqrt(5)]
    for filename in test_filenames:
        with h5py.File(filename, 'r') as f:
            real_part = np.array(f['Hset']['real'])
            imag_part = np.array(f['Hset']['imag'])
            H = real_part + 1j * imag_part
            Hset = torch.from_numpy(H[0:test_number])
            Hset = Hset * 10 ** (Ptot / 20) #* testfcset[m] / 3.5e9
            test_H.append(Hset)
        # m = m+1


    ##DNN hyper-parameters
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    MAX_EPOCH = 1
    epoch2equal = 0  # before which epoch, the transmit power is equally allocated to all users to avoid a local minimum that only one user is served
    BS = min(200, train_number) #batch size
    LEARNING_RATE = 0

    ##select  DNN
    y_layer = [128]*6
    y_net = AGNN(input_dim=2, hidden_dim=y_layer, output_dim=4 * NRF, is_attention=True, NRF=NRF)

    # y_layer = [256]*6
    # y_net = GNN3D(input_dim=2, hidden_dim=y_layer, output_dim=4*NRF, NRF=NRF)

    # y_layer = [256, 256, 256, 256, 16]
    # y_net = CNN(input_dim=2, hidden_dim=y_layer, output_dim=2 * K * NRF + 2 * NRF * NT, NRF=NRF)

    y_net.to(device)
    # y_optimizer = torch.optim.Adam([{'params': y_net.parameters()}], lr=LEARNING_RATE, weight_decay=0)
    # y_scheduler = torch.optim.lr_scheduler.MultiStepLR(y_optimizer, [30, 70, 110, 150], gamma=0.3, last_epoch=-1)

    total_params = sum(p.numel() for p in y_net.parameters())
    timestr = time.strftime("%m%d%H%M%S", time.localtime())
    print('K', K, 'NT', NT, 'NRF', NRF, 'Ptot', Ptot, 'sigma2', sigma2, 'time', timestr)
    print('train_number', train_number, 'test_number', test_number, 'batch_size', BS, 'ini_learning_rate', LEARNING_RATE, 'y_layer', y_layer, 'epoch2equal', epoch2equal, 'trainable_weights', total_params)
    # print(train_filenames)
    print(test_filenames)

    ## UAT hyper-parameters
    is_UAT = True  # for loading a well-trained model, not actually using UAT
    T_UAT = 5
    alpha_UAT = 0.2

    epoch_UAT = 9999  # don't change
    is_deH = False  # don't change

    if is_UAT:
        # load a half-trained (by conventional unsupervised learning for E_A=50 epoch's) DNN and train it with UAT
        # address = 'model50_' + '1031161644' + '_K' + str(K) + '_NT' + str(NT) + '.ckp'
        address = 'model_' + '1113093816' + '_K' + str(K) + '_NT' + str(NT) + '.ckp'

        y_net.load_state_dict(torch.load(address)['y_net'])
        # y_optimizer.load_state_dict(torch.load(address)['y_optimizer'])
        # y_scheduler.load_state_dict(torch.load(address)['y_scheduler'])
        epoch2equal = 0
        epoch_UAT = 0
        # print('UAT', 'T', T_UAT, 'alpha', alpha_UAT, 'load', address, epoch_UAT, epoch2equal)
        print('test', address)



    ## test
    sys.stdout.flush()
    # highestSE = 0

    for epoch in range(MAX_EPOCH):

        if epoch % 1 == 0:
            y_net.eval()

            with torch.no_grad():
                sum_train_loss = [0]*len(train_filenames)
                sum_test_loss = [0]*len(test_filenames)

                for bb in range(int(test_number / BS)):
                    for m in range(len(train_filenames)):
                        batch_x = train_H[m][bb * BS:(bb + 1) * BS].to(device)
                        SNR = torch.sum(torch.abs(batch_x) ** 2, dim=[1, 2], keepdim=True) / sigma2  # BS,1,1
                        sqrtSNR = torch.sqrt(SNR)
                        batch_x = batch_x / sqrtSNR
                        y_pred, z_pred = y_net(batch_x, sqrtSNR=sqrtSNR, equal_user=False)
                        SE = cal_sumrate(batch_x * sqrtSNR, y_pred, z_pred, sigma2=sigma2, device=device)
                        sum_train_loss[m] = sum_train_loss[m] + SE.detach().cpu().numpy()

                    for m in range(len(test_filenames)):
                        batch_x = test_H[m][bb * BS:(bb + 1) * BS].to(device)
                        SNR = torch.sum(torch.abs(batch_x) ** 2, dim=[1, 2], keepdim=True) / sigma2  # BS,1,1
                        sqrtSNR = torch.sqrt(SNR)
                        batch_x = batch_x / sqrtSNR
                        y_pred, z_pred = y_net(batch_x, sqrtSNR=sqrtSNR, equal_user=False)
                        SE = cal_sumrate(batch_x * sqrtSNR, y_pred, z_pred, sigma2=sigma2, device=device)
                        sum_test_loss[m] = sum_test_loss[m] + SE.detach().cpu().numpy()


                time_end = time.time()
                print(epoch, "{:.2f}".format(time_end - time_start), end=' ')
                for m in range(len(train_filenames)):
                    print(f"{sum_train_loss[m] / (test_number / BS):.4f}", end=' ')
                print('|', end=' ')
                for m in range(len(test_filenames)):
                    print(f"{sum_test_loss[m] / (test_number / BS):.4f}", end=' ')
                print()
                sys.stdout.flush()
                time_start = time.time()



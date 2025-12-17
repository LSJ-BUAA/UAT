import deepmimo as dm
import numpy as np
import math
import h5py


# ===== PARAMETER START =====
params = dm.ChannelParameters()

# Configure BS antenna array
params.bs_antenna.shape = [4, 1]  # array
params.bs_antenna.spacing = 0.5  # Half-wavelength spacing

# Configure UE antenna array
params.ue_antenna.shape = [1, 1]  # Single antenna
params.ue_antenna.spacing = 0.5

# Configure OFDM parameters
params.ofdm.subcarriers = 1667  # Number of subcarriers 100MHz/60kHz, for fc > 6 GHz
params.ofdm.bandwidth = 100e6  # 100 MHz bandwidth

# params.ofdm.subcarriers = 1333  # Number of subcarriers 20MHz/15kHz, for fc < 6 GHz
# params.ofdm.bandwidth = 20e6  # 20 MHz bandwidth

params.ofdm.selected_subcarriers = [0]  # Which subcarriers to generate

# Generate frequency-domain channels
params.doppler = False
params.freq_domain = True
# ===== PARAMETER END =====


scenario = 'o1_28'
# dm.summary(scenario)
# BSset = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # the available BSs in this scenario dataset
BSset = [3,4,5,6,7,8,9,10,11,12,13,14]  # the available BSs in this scenario dataset

# UEset = [0, 1, 2]  # the available user sets in this scenario dataset
UEset = [0]  # the available user sets in this scenario dataset

number = 2000  # number of samples
K = 10  # number of users

possibleUEpos_sampled = 497931
UEpos = np.array([np.random.choice(possibleUEpos_sampled, K, replace=False) for _ in range(number)])

L = len(BSset)
n_tx = math.prod(params.bs_antenna.shape)
n_rx = math.prod(params.ue_antenna.shape)
n_subcarrier = len(params.ofdm.selected_subcarriers)

n_BSset = len(BSset)
n_UEset = len(UEset)

channelSet = np.zeros((number, L, K, n_tx), dtype=np.complex128)

# thre = [0, 497931, 497931+199281, 497931 + 199281 + 487711]
thre = [0, 497931]
for l in range(n_BSset):
    for j in range(len(UEset)):
        dataset = dm.load(scenario, tx_sets=[BSset[l]], rx_sets=[UEset[j]])
        active_idxs = dataset.get_active_idxs()
        print(active_idxs)
        dataset_t = dataset.subset(active_idxs)
        channels = dataset_t.compute_channels(params)
        print(channels.shape)

        for n in range(number):
            for k in range(K):
                if thre[j]<=UEpos[n][k]<thre[j+1]:
                    idx = UEpos[n][k]-thre[j+1]
                    channelSet[n,l,k,:] = channels[idx].reshape(n_tx)


noiseFigure = 9
noiseVariancedBm = -174 + 10 * math.log10(params.ofdm.bandwidth) + noiseFigure - 30  # in [dBW]
channelSet = channelSet * 10 ** (-noiseVariancedBm / 20)  # scale up the channel coefficients to avoid precision loss

name_add = '../datasets/CH' + scenario + '_NT' + str(n_tx) + '_K' + str(K) + '_L' + str(L) + '_num' + str(number)
with h5py.File(name_add + ".mat", "w") as f:
    f.create_dataset("data", data=channelSet)
print(channelSet.shape)
print(channelSet)




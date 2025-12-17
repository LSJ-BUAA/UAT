import deepmimo as dm
import numpy as np
import math
import h5py

# DeepMIMO 4.0.0 beta


def split(n, k):
    base = n // k
    remainder = n % k
    group = [base] * (k - 1) + [base + remainder]
    return group


# ===== PARAMETER START =====
params = dm.ChannelParameters()

# Configure BS antenna array
params.bs_antenna.shape = [8, 2]  # array
params.bs_antenna.spacing = 0.5  # Half-wavelength spacing

# Configure UE antenna array
params.ue_antenna.shape = [1, 1]  # Single antenna
params.ue_antenna.spacing = 0.5

# Configure OFDM parameters
# params.ofdm.subcarriers = 1667  # Number of subcarriers 100MHz/60kHz, for fc > 6 GHz
# params.ofdm.bandwidth = 100e6  # 100 MHz bandwidth

params.ofdm.subcarriers = 1333  # Number of subcarriers 20MHz/15kHz, for fc < 6 GHz
params.ofdm.bandwidth = 20e6  # 20 MHz bandwidth

params.ofdm.selected_subcarriers = [0]  # Which subcarriers to generate

# Generate frequency-domain channels
params.doppler = False
params.freq_domain = True
# ===== PARAMETER END =====

scenario = 'o1_3p5'
BSset = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # the available BSs in this scenario dataset
UEset = [0, 1, 2]  # the available user sets in this scenario dataset

number = 2000  # number of samples
K = 4  # number of users

n_tx = math.prod(params.bs_antenna.shape)
n_rx = math.prod(params.ue_antenna.shape)
n_subcarrier = len(params.ofdm.selected_subcarriers)

n_BSset = len(BSset)
n_UEset = len(UEset)

groups = split(number, n_BSset)
groups = [split(x, n_UEset) for x in groups]  # the 2000 samples come from each pair of elements from BSset and UEset, such as [3]-[0], [3]-[1], [3]-[2], [4]-[0]...

channelSet = np.zeros((number, K, n_tx), dtype=np.complex128)

count = 0
for i in range(n_BSset):
    for j in range(n_UEset):
        dataset = dm.load(scenario, tx_sets=[BSset[i]], rx_sets=[UEset[j]])

        active_idxs = dataset.get_active_idxs()
        dataset_t = dataset.subset(active_idxs)

        channels = dataset_t.compute_channels(params)

        n_temp = groups[i][j]
        # idx = np.random.choice(channels.shape[0], size=n_temp*K, replace=True)  # if the number of available users < n_temp*K
        idx = np.random.choice(channels.shape[0], size=n_temp*K, replace=False)
        channels_subset = channels[idx]
        channels_subset = channels_subset.reshape(n_temp, K, n_tx)

        channelSet[count:count+n_temp] = channels_subset
        count = count + n_temp

noiseFigure = 9
noiseVariancedBm = -174 + 10 * math.log10(params.ofdm.bandwidth) + noiseFigure  # in [dBm]
print(noiseVariancedBm)
channelSet = channelSet * 10 ** (-noiseVariancedBm / 20)  # scale up the channel coefficients to avoid precision loss

name_add = '../datasets/CH' + scenario + '_NT' + str(n_tx) + '_K' + str(K) + '_num' + str(number)
# np.save(name_add + ".npy", channelSet)
with h5py.File(name_add + ".mat", "w") as f:
    f.create_dataset("data", data=channelSet)

print(channelSet.shape)




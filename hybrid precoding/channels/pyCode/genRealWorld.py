import numpy as np
import h5py
import random

# Ultra Dense Indoor MaMIMO CSI Dataset: URA_lab_LoS
# https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

K = 4
NT = 64
number = 2000

Dset = np.zeros((number,K,NT), dtype=np.complex64)

for n in range(number):
    if n % 100 == 0:
        print(n)
    for k in range(K):
        s = str(random.randint(0, 252003)).zfill(6)
        value = np.load('./URA_lab_LoS/samples/channel_measurement_'+s+'.npy')
        Dset[n, k, :] = value[:, 50]

with h5py.File("../datasets/CHindoor_NT"+str(NT)+"_K4_num2000.mat", "w") as f:
    f.create_dataset("data", data=Dset)



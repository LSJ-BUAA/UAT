import numpy as np
import h5py
import random

# Ultra Dense Indoor MaMIMO CSI Dataset: DIS_lab_LoS
# https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

K = 10
NT = 8
L = 8
number = 2000

Dset = np.zeros((number,K,L*NT), dtype=np.complex64)

for n in range(number):
    if n % 100 == 0:
        print(n)
    for k in range(K):
        s = str(random.randint(0, 252003)).zfill(6)
        value = np.load('./DIS_lab_LoS/samples/channel_measurement_'+s+'.npy')
        Dset[n, k, :] = value[:, 50]

Dset = Dset.reshape(number,K,L,NT)
Dset = Dset.transpose(0,2,1,3) # number, L, K, NT

print(Dset.shape)

with h5py.File("../datasets/CHindoor_NT"+str(NT)+"_K"+str(K)+"_L"+str(L)+"_num2000.mat", "w") as f:
    f.create_dataset("data", data=Dset)



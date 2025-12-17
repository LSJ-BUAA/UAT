clear
clc
close all

tic
%% settings
K = 10; % number of single-antenna users
NT = 25; % number of AP antennas
L = 4; % number of APs

D = 500; % side length of the square area, in [m]
% ty = 1; %1 for UMi, 2 for UMa, 4 for Mixed Office, 5 for Open Office

%% 
Time = 2000; %number of samples in the dataset

Hset = zeros(NT,K,L,Time,'single');

parfor t = 1:Time
    if mod(t,100) == 0
        fprintf('time %d\n', t);
    end
    
    %% select channel model
    
    H = genCHleng(NT, K, L, D);
    % H = genCHtype(NT, K, L, ty);
    
    %% 
    H = single(H);
    Hset(:,:,:,t) = H;
    
end

%% save the dataset

% the filename should be manually modified with the channel model
filename = ['../datasets/CHleng_NT' num2str(NT) '_K' num2str(K) '_L' num2str(L) '_D' num2str(D) '_num' num2str(Time) '.mat']; 
% filename = ['../datasets/CHty_NT' num2str(NT) '_K' num2str(K) '_L' num2str(L) '_Ty' num2str(ty) '_num' num2str(Time) '.mat']; 
save(filename, 'Hset')
toc

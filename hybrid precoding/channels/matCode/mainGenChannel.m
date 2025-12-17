close all
clear
clc

%% settings
K = 4; % number of single-antenna users
NT = 16; % number of BS antennas

fc = 3.5e9;

% d_lam = 0.5;

% r =5;

% type = 1; %1 for UMi, 2 for UMa, 4 for Mixed_Office, 5 for Open_Office

%% 
Time = 2000; %number of samples in the dataset

Hset = zeros(NT,K,Time,'single');

parfor t = 1:Time
    if mod(t,100) == 0
        fprintf('time %d\n', t);
    end

    %% select channel model

    H = genCHfc(K,NT,fc);

    % H = genCHspacing(K,NT,d_lam);

    % H = genCHhotspot(K,NT,r);
    
    % H = genCHtype(K,NT,type);

    %% single precision is enough
    H = single(H);
    Hset(:,:,t) = H;
    
end

%% save the dataset

filename = ['../datasets/CHfc_NT' num2str(NT) '_K' num2str(K) '_fc' num2str(fc) '_num' num2str(Time) '.mat']; 

% filename = ['../datasets/CHspacing_NT' num2str(NT) '_K' num2str(K) '_d' num2str(d_lam) '_num' num2str(Time) '.mat']; 
% 
% filename = ['../datasets/CHhotspot_NT' num2str(NT) '_K' num2str(K) '_r' num2str(r)  '_num' num2str(Time) '.mat']; 
% 
% filename = ['../datasets/CHtype_NT' num2str(NT) '_K' num2str(K) '_Ty' num2str(type)  '_num' num2str(Time) '.mat']; 

save(filename, 'Hset')

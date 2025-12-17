clear
clc
close all

tic
%% Result format
% MRT+equal | ZF+equal | RZF+equal
% MRT+SCA   | ZF+SCA    | RZF+SCA
% MRT+APG   | ZF+APG   | RZF+APG

%% settings
K = 10; % number of users
NT = 4; % number of AP antennas
L = 25; % number of APs

D = 500; % in [m]

PmaxdBm = 20+35.3*log10(D/500);
Pmax = 10^((PmaxdBm-30)/10); % maximal power of each AP, in [Watt]

SEqos = 1*ones(K,1); %QoS for each user, in [bps/Hz]

Pfix = L*(NT*0.2+0.825); % fixed power consumption, in [Watt]
a = 1/0.4; % reciprocal of amplifier efficiency
b = 0.25e-3; % coefficient for traffic power consumption, in [Watt/Mbps]

B = 20; % bandwidth, in [MHz]

sigma2 = 1; % dont change, this has been equivalently represented in large-scale fadings

%% load dataset

load(['../channels/datasets/CH' 'leng' '_NT' num2str(NT) '_K' num2str(K) '_L' num2str(L) '_D' num2str(D) '_num' num2str(2000) '.mat']) % load a Hset

% type = 1; % 1 for UMi, 2 for UMa, 4 for mixed office, 5 for open office
% load(['../channels/datasets/CH' 'ty' '_NT' num2str(NT) '_K' num2str(K) '_L' num2str(L) '_Ty' num2str(type) '_num' num2str(2000) '.mat'])

% data = h5read('../channels/datasets/CHo1_28_NT4_K10_L12_num2000.mat', '/data');
% Hset = data.r + 1j*data.i;

%% 
Time = 2000;

EE_Table = zeros(9,Time);

parfor t = 1:Time

    if mod(t,100) == 0
        fprintf('time %d\n', t);
    end

    EE_Table_temp = zeros(9,1);

    %% channel
    % H = (randn(NT,K,L) + 1j*randn(NT,K,L))/sqrt(2); % Rayleigh 
    
    H = double(Hset(:,:,:,t));

    %% beamforming 1: MRT

    W = H; 

    W = W./(sqrt(sum(abs(W).^2,1))); %normalize
    G = pagemtimes(pagectranspose(H),W);

    %% power allocation methods
    %% 11 equal
    P = Pmax/K*ones(L,K);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(1,1) = EE;

    %% 12 SCA
    P = powerSCA(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(2,1) = EE;

    %% 13 APG
    P = powerAPG(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(3,1) = EE;

    %% beamforming 2: ZF

    if NT<K 
        W = pagemtimes(pageinv(pagemtimes(H,pagectranspose(H))),H);%ZF
    else
        W = pagemtimes(H,pageinv(pagemtimes(pagectranspose(H),H)));%ZF
    end

    W = W./(sqrt(sum(abs(W).^2,1))); %normalize
    G = pagemtimes(pagectranspose(H),W);

    %% power allocation methods
    %% 21 equal
    P = Pmax/K*ones(L,K);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(4,1) = EE;

    %% 22 SCA
    P = powerSCA(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(5,1) = EE;

    %% 23 APG
    P = powerAPG(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(6,1) = EE;

    %% beamforming 3: RZF
    if NT<K 
        W = pagemtimes(pageinv(pagemtimes(H,pagectranspose(H)) + K*sigma2/Pmax*eye(NT)),H);%RZF
    else
        W = pagemtimes(H,pageinv(pagemtimes(pagectranspose(H),H) + K*sigma2/Pmax*eye(K)));%RZF
    end
    
    W = W./(sqrt(sum(abs(W).^2,1))); %normalize
    G = pagemtimes(pagectranspose(H),W);
    
    %% power allocation methods
    %% 31 equal
    P = Pmax/K*ones(L,K);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(7,1) = EE;
    
    %% 32 SCA
    P = powerSCA(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(8,1) = EE;
    
    %% 33 APG
    P = powerAPG(G,sigma2,Pfix,a,b,B,Pmax,SEqos);
    [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
    EE_Table_temp(9,1) = EE;

    %% END
    EE_Table(:,t) = EE_Table_temp;

end

nan_ratio = mean(isnan(EE_Table), 2);

EE_Table(isnan(EE_Table)) = 0; 
ResultTable = mean(EE_Table,2);

toc
AEEresult = reshape(ResultTable,3,3) % in [Mbits/Joule]
VRresult = reshape(nan_ratio,3,3)*100 % in percent


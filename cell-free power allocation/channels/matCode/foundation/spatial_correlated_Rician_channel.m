% cell-free downlink channels
% ref. Spectral and Energy Efficiency in Cell-Free Massive MIMO Systems Over Correlated Rician Fading, but parameters differ
% output is H with size [NT,K,L], normlized by noise power
close all
clear
clc

tic
%% settings
K = 10; % number of single-antenna users
L = 25; % number of APs
NT = 4; % number of antennas at each AP

squareLength = 500; % side length of the area, in [m]
APheight = 10; % height of APs, in [m]
UEheight = 1.5; % height of users, in [m]

fc = 3.5e9; % carrier freq., in [Hz]
B = 20e6; % bandwidth, in [Hz]
d_lam = 0.5; % spacing of antennas, normlized by lambda, for LOS
r = 0; %spatial correlation coefficient of antennas, for NLOS
c = 3e8; % speed of light, in [m/s]

allowLOS = true;

dbp = 4*(APheight-1)*(UEheight-1)*fc/c;
orderNT = (0:NT-1)';

%% locations
APpositions = (rand(L,1) + 1i*rand(L,1)) * squareLength;
UEpositions = (rand(K,1) + 1i*rand(K,1)) * squareLength;

% wrapped AP locations using wrap around
wrapHorizontal = repmat([-squareLength 0 squareLength],[3 1]);
wrapVertical = wrapHorizontal';
wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
APpositionsWrapped = repmat(APpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[L 1]);

%% generate channels
noiseFigure = 9;
noiseVariancedBW = -174 + 10*log10(B) + noiseFigure - 30; % in [dBW]

H = zeros(NT,K,L);
R = toeplitz(r.^(0:NT-1));
Rsqrt = sqrtm(R);

for k = 1:K
    [distancetoUE,~] = min(abs(APpositionsWrapped - repmat(UEpositions(k),size(APpositionsWrapped))),[],2); % the 2d distances from every AP to a UE
    distances = sqrt((APheight-UEheight)^2+distancetoUE.^2); % the 3d distances
    
    PrLOS = 18./distancetoUE + exp(-distancetoUE/36) .* (1-18./distancetoUE); % probability of LOS
    
    for l =1:L
        d2d = distancetoUE(l);
        d3d = distances(l);

        h_NLOS =  (randn(NT,1) + 1j*randn(NT,1))/sqrt(2);
        h_NLOS = Rsqrt*h_NLOS; 

        if d2d <= dbp
            PL_LOS = 32.4 + 21*log10(d3d) + 20*log10(fc/1e9);
        else
            PL_LOS = 32.4 + 40*log10(d3d) + 20*log10(fc/1e9) - 9.5*log10(dbp^2+(APheight-UEheight)^2);
        end
        
        if allowLOS && rand < PrLOS(l)
            % LOS
            theta_t = 2*pi*rand();
            h_LOS = exp(-1j*2*pi*d_lam * orderNT * sin(theta_t));
            
            sigma_sf = 4;
            PL = PL_LOS;
            kRic_dB = 9 + 5*randn();
            kRic = 10^(kRic_dB/10);
            
            beta_dB = - PL + sigma_sf*randn() - noiseVariancedBW;
            beta = 10^(beta_dB/20);
            H(:,k,l) = beta*(sqrt(kRic/(1+kRic))*h_LOS + sqrt(1/(1+kRic))*h_NLOS);
            
        else
            %NLOS
            sigma_sf = 7.82;
            PL_NLOS = 35.3*log10(d3d) + 22.4 + 21.3*log10(fc/1e9) - 0.3*(UEheight-1.5);
            PL = max(PL_NLOS,PL_LOS);
            
            beta_dB = - PL + sigma_sf*randn() - noiseVariancedBW;
            beta = 10^(beta_dB/20);
            H(:,k,l) = beta*h_NLOS;
        end
    end
end

size(H)
sum(abs(H).^2,"all")
toc

function [ FRF,FBB ] = SVDp( H, NRF,SNR)
    %SVD+RZF
    sigma2=10^(-SNR/10);
    [K,~] = size(H);
    
    [U,~,~] = svd(H');
    FRF = U(:,1:NRF);
    FRF = FRF./abs(FRF);
    
    H = H*FRF;
    
    FBB = H'*((H*H'+K*sigma2*eye(K))\eye(K));
    FBB =  FBB./sqrt(sum(sum(abs(FRF*FBB).^2)));
end
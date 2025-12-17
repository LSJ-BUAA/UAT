function [ FRF,FBB ] = PEM( Fopt, NRF )
    % phase extraction
    [Nt, Ns] = size(Fopt);
    mynorm = [];
    FRF = exp( 1i * unifrnd (0,2*pi,Nt,NRF) );
        while (isempty(mynorm) || abs( mynorm(end) - mynorm(end-1) ) > 1e-10)
            [U,S,V] = svd(Fopt'*FRF);
            FBB = V(:,[1:Ns])*U';
            
            mynorm = [mynorm, norm(Fopt * FBB' - FRF,'fro')^2];
            FRF = exp(1i * angle(Fopt * FBB'));
            mynorm = [mynorm, norm(Fopt * FBB' - FRF,'fro')^2];
        end
    
    FBB = FBB / norm(FRF * FBB,'fro');

end
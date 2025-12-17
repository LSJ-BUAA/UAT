function [ FRF,FBB ] = MO( Fopt, NRF )
    %manifold optimization needs toolbox Manopt
    [Nt, ~] = size(Fopt);
    y = [];
    FRF = exp( 1i*unifrnd(0,2*pi,Nt,NRF) );
    
    iter = 0;
    while(isempty(y) || abs(y(1)-y(2))>1e-5)
        FBB = pinv(FRF) * Fopt;
        y(1) = norm(Fopt - FRF * FBB,'fro')^2;
        [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
        iter = iter + 1;

        if iter > 1000
            break
        end

    end

    FBB = FBB / norm(FRF * FBB,'fro');
end
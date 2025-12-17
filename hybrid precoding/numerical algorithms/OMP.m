function [ FRF, FBB ] = OMP( Fopt, NRF, At )
    % orthogonal matching pursuit
    FRF = [];
    Fres = Fopt;
        for k = 1:NRF
            PU = At' * Fres;
            [~,bb] = max(sum( abs(PU).^2, 2 ));
            FRF = [FRF , At(:,bb)];
            FBB = pinv(FRF) * Fopt;
            Fres = (Fopt - FRF * FBB) / norm(Fopt - FRF * FBB,'fro');
        end
    FBB = FBB / norm(FRF * FBB,'fro');
end


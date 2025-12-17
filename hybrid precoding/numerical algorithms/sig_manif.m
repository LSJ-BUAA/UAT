function [y, cost] = sig_manif(Fopt, FRF, FBB)
    [Nt, NRF] = size(FRF);
    K = size(FBB,3);
    
    manifold = complexcirclefactory(Nt*NRF);
    problem.M = manifold;
    
    for k = 1:K    
        temp = Fopt(:,:,k);
        A = kron(FBB(:,:,k).', eye(Nt));
        C1(:,:,k) = temp(:)'*A;
        C2(:,k) = A'*temp(:);
        C3(:,:,k) = A'*A;
        C4(k) = norm(temp,'fro')^2;
    end
    B1 = sum(C1,3);
    B2 = sum(C2,2);
    B3 = sum(C3,3);
    B4 = sum(C4);
    
    problem.cost = @(x) -B1*x - x'*B2 + trace(B3*x*x') + B4;
    problem.egrad = @(x) -2*B2 + 2*B3*x;
    
    % checkgradient(problem);
    warning('off', 'manopt:getHessian:approx');
    options.verbosity = 0;
    [x,cost,info,options] = conjugategradient(problem, FRF(:), options);
    % [x,cost,info,options] = trustregions(problem, FRF(:));
    y = reshape(x,Nt,NRF);

end
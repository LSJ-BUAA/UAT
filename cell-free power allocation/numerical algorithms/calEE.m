function [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos)
    % calculate EE and SE
    %G[K,I,L] P[L,I]
    [K, ~, L] = size(G);
    I = K;
    
    %% check
    if any(P < 0)
        warning('some p < 0')
    end
    
    if any(sum(P, 2) > Pmax + 1e-3)
        warning('some AP > Pmax')
    end
    
    Psum = sum(P,'all');
    
    %% SE
    C = sqrt(P);
    G = permute(G, [1, 3, 2]); %[K,L,I]
    C = reshape(C,[L,1,I]); %[L,1,I]
    GC = reshape(pagemtimes(G,C), [K,I]); %[K,I]
    
    GC2 = abs(GC).^2; %[K,I]
    
    SEperUE = log2(1+diag(GC2)./(sum(GC2,2)-diag(GC2)+sigma2));
    
    SE = sum(SEperUE);
    
    %% EE
    
    Ptotal = Pfix + a*Psum + b*B*SE;
    EE = B*SE/Ptotal;
    
    if any(SEperUE < SEqos-1e-3)
        % warning('some SE < SEqos')
        EE = NaN;
    end

end
function out = PenalC(xi,G,C,sigma2,Pfix,a,SEqos)
    [L, K] = size(C);
    I = K;

    G = permute(G, [1, 3, 2]); %[K,L,I]
    C = reshape(C,[L,1,I]); %[L,1,I]
    GC = reshape(pagemtimes(G,C), [K,I]); %[K,I]
    GC2 = abs(GC).^2; %[K,I]
    INR = sum(GC2,2)-diag(GC2)+sigma2; %[K,1]
    % SINR = diag(GC2)./INR; %[K,1]
    % SEperUE = log2(1 + SINR);  %[K,1]
    % SE = sum(SEperUE);
    
    % u = SE;
    % v = Pfix + a*sum(C.^2,'all');
    
    % Phik = (max(0,sqrt((2.^(SEqos)-1) .* INR) - abs(diag(GC)))).^2;  %[K,1]
    Phik = (max(0,(2.^(SEqos)-1).* INR - diag(GC2)));  %[K,1]
    out = sum(Phik);

end
function nabla = nabla_fxiC(xi,G,C,sigma2,Pfix,a,SEqos)
    [L, K] = size(C);
    I = K;
    
    Gt = permute(G, [1, 3, 2]); %[K,L,I]
    Ct = reshape(C,[L,1,I]); %[L,1,I]
    GC = reshape(pagemtimes(Gt,Ct), [K,I]); %[K,I]
    GC2 = abs(GC).^2; %[K,I]
    INR = sum(GC2,2)-diag(GC2)+sigma2; %[K,1]
    SINR = diag(GC2)./INR; %[K,1]
    SEperUE = log2(1 + SINR);  %[K,1]
    SE = sum(SEperUE);
    
    u = SE;
    v = Pfix + a*sum(C.^2,'all');
    
    sqrt_star = sqrt((2.^(SEqos)-1) .* INR);

    % Phik = (max(0, sqrt_star - diag(GC))).^2;
    % Phi = sum(Phik);
    % out = u/v - xi*Phi;

    %% cal gradient

    G = permute(G, [3, 1, 2]);  %[L,K,I]

    duk = zeros(L,K,K);
    for k=1:K
        for i = 1:I
            if i == k
                duk(:,i,k) = 2/log(2)/ (1+SINR(k)) * (real(G(:,k,i)*G(:,k,i)')*C(:,i)/INR(k)); 
            else %i \neq k
                duk(:,i,k) = -2/log(2)/ (1+SINR(k)) * (GC2(k)*real(G(:,k,i)*G(:,k,i)')*C(:,i)/(INR(k)^2)); 
            end
        end
    end
    du = sum(duk,3);

    dv = 2*a*C;
    
    absdiagGC = abs(diag(GC));
    gk = sqrt_star - absdiagGC;
    dPhik = zeros(L,K,K);
    for k = 1:K
        for i=1:I
            if i == k
                dPhik(:,i,k) = -2*max(0,gk(k))*(real(G(:,k,i)*G(:,k,i)')*C(:,i))/absdiagGC(k); 
            else %i \neq k
                dPhik(:,i,k) = 2*max(0,gk(k))*(2^SEqos(k)-1)/sqrt_star(k)*real(G(:,k,i)*G(:,k,i)')*C(:,i); 
            end
        end
    end
    dPhi = sum(dPhik,3);

    nabla = (du*v-dv*u)/v^2 - xi*dPhi;
end
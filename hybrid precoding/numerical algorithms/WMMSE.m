function [V] = WMMSE( H,SNR)
    epsilon = 1e-5;
    sigma2=10^(-SNR/10);
    [K,N] = size(H);
    
    
    V = H'*((H*H'+K*sigma2*eye(K))\eye(K)); %initialized by RZF
    % V = ones(N,K); %initialized by ones
    % V = randn(N,K)+1j*randn(N,K); %initialized by random
    
    V = V/sqrt(sum(sum(abs(V).^2))); %satisfy power constraint
    % V = V./sqrt(sum(abs(V).^2))/sqrt(K); %equal power to every user
    
    U = zeros(1,K);
    W = zeros(1,K);
    Wre = W+100;
    it = 0;
    while abs(sum(log2(abs(W)))-sum(log2(abs(Wre))))>epsilon
        Wre = W;
        for m = 1:K
            temp = 0;
            for i = 1:K
                temp = temp + H(m,:)*V(:,i)*V(:,i)'*H(m,:)';
            end
            U(m) = 1/(temp+sigma2)*H(m,:)*V(:,m);
            W(m) = 1/(1-U(m)'*H(m,:)*V(:,m));
        end
        
        %% calculate mu
        S = 0;
        for k=1:K
               S = S + H(k,:)'*U(k)*W(k)*U(k)'*H(k,:);
        end
        [D,L] = eig(S);
        S2 = 0;
        for k=1:K
               S2 = S2 + H(k,:)'*U(k)*W(k)*W(k)'*U(k)'*H(k,:);
        end
        P = D'*S2*D;
        P = diag(abs(P));
        L = diag(abs(L));
        
        a = 0; % find mu s.t. sum(P./((L+mu).^2))-1==0
        b = 100;
        ind = b-a;
        while ind > epsilon
            xx = (a+b)/2;
            if (sum(P./((L+a).^2))-1)*(sum(P./((L+xx).^2))-1) < 0
                b = xx;
            else
                a = xx;
            end
            ind = b - a;
        end
        mu = xx;
        
        %% update V
        for m=1:K
           V(:,m) =(S+mu*eye(N))\H(m,:)'*U(m)*W(m);
        end
        
        it = it + 1;
        if it >1000
            break
        end
    end

end
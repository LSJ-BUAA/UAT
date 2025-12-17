function P = powerSCA(G,sigma2,Pfix,a,b,B,Pmax,SEqos)
%need toolbox yalmip and mosek
    [K,I,L]=size(G);
    sigma = sqrt(sigma2);
    %% SCA initial
    opts=sdpsettings('solver','mosek','verbose',0,'dualize',0);
    C_var = sdpvar(L,K,'full');
    F = [C_var(:)>=0];
    F = [F, cone([sqrt(Pmax)*ones(1,L);C_var'])]; %power constraint
    
    for k=1:K
        GCk=[];
        for i=1:K
            GCk = [GCk; reshape(G(k,i,:),1,L)*C_var(:,i)];
        end
        F = [F, cone([GCk([1:k-1, k+1:end]); sigma], (GCk(k))/sqrt(2^SEqos(k,1)-1))]; %QoS constraint
    end
    
    diagnotics = optimize(F,[],opts); % solve feasibility problem
    if diagnotics.problem==0 % if feasible
        C_n = value(C_var);
        P = C_n.^2;
        [~,~,~,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos);
        u_n = 2.^SEperUE;
        % u_n = 2.^SEqos;
    else
        % warning('infeasible')
        P = zeros(L,K);
        return
    end
    
    % P = value(C_var).^2;
    % [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos)
    
    %% SCA iteration
    maxIteration = 30;
    eps = 1e-3;
    termination_record = Inf;
    
    C_var = sdpvar(L,K,'full') ;
    t_var = sdpvar(K,1);
    u_var = sdpvar(K,1);
    theta_var = sdpvar;
    
    obj = sum(t_var);
    opts=sdpsettings('solver','mosek','verbose',0,'dualize',0);
    
    for iIter=1:maxIteration 
        F=[];
        
        F = [F,t_var(:)>=0];
        F = [F,C_var(:)>=0];
        F = [F,u_var(:)>=0];
        F = [F, theta_var>=0];
        
        F = [F, cone([theta_var*sqrt(Pmax)*ones(1,L); C_var'])]; %power constraint
        
        F = [F,cone([sqrt(Pfix)*theta_var; sqrt(a)*C_var(:); 0.5*(theta_var-1)], 0.5*(theta_var+1))]; 
        
        F = [F,cone([(u_var+theta_var*(log(u_n)+1)-log(2)*t_var)';   (u_var-theta_var*(log(u_n)+1)+log(2)*t_var)';   2*sqrt(u_n)'*theta_var])]; 
        
        
        GC = reshape(pagemtimes(permute(G, [1, 3, 2]), reshape(C_n,[L,1,K])), [K,K]); % for compute fvalue
        GC2 = abs(GC).^2;  % for compute fvalue
    
        for k=1:K
            GCk=[];
            for i=1:K
                GCk = [GCk; reshape(G(k,i,:),1,L)*C_var(:,i)];
            end
            F = [F, cone([GCk([1:k-1, k+1:end]); theta_var*sigma], (GCk(k))/sqrt(2^SEqos(k,1)-1))]; %qos constraint (32b)
            
            %% compute F(-,-;-,-)
            i=1;
            g_ki = reshape(G(k,i,:),L,1);
            S = C_n(:,i)'*g_ki*g_ki'*(C_var(:,i)-theta_var*C_n(:,i));
            for i=2:K
                g_ki = reshape(G(k,i,:),L,1);
                S = S+C_n(:,i)'*g_ki*g_ki'*(C_var(:,i)-theta_var*C_n(:,i));
            end
    
            fvalue = (sum(GC2(k,:))+sigma2)/u_n(k);
            Fvalue = theta_var*fvalue - fvalue*(u_var(k)-theta_var*u_n(k))/u_n(k) + 2/u_n(k)*S;
    
            F = [F; cone([2*GCk([1:k-1, k+1:end]); 2*theta_var*sigma; theta_var-Fvalue], theta_var+Fvalue)]; 
        end
    
        %% slove
        diagnotics = optimize(F,-obj,opts);
        if (diagnotics.problem==0) 
            u_n = value(u_var/theta_var); %update
            C_n = value(C_var/theta_var); 
        else
            % warning('SCA is down')
            break
        end
    
        %% termination condition
        if abs(sum(value(t_var))-termination_record)<eps
            break
        else
            termination_record = sum(value(t_var));
        end
    
    end
    
    %% performance
    P = value(C_var/theta_var).^2;
    % [EE,SE,Psum,SEperUE] = calEE(G,P,sigma2,Pfix,a,b,B,Pmax,SEqos)
end
function P = powerAPG(G,sigma2,Pfix,a,b,B,Pmax,SEqos)
    [K,I,L]=size(G);
    % sigma = sqrt(sigma2);

    P = Pmax/K*ones(L,K); %equal
    %% APG initial
    xi = 0.1;
    
    C_prev = sqrt(P);
    C_now = C_prev;
    
    z = 1.1*C_prev;
    y_prev = C_prev;
    v_prev = 1.1 * C_prev;
    
    t_prev=1;
    t_now=1;
    
    nu = 0.5;
    delta = 1e-5;
    
    interval = 10;
    
    maxInner = 200;
    maxOuter = 10;
    maxLS = 20;%for line search iter
    
    bestobj = zeros(maxInner*maxOuter,1);
    m = 1;
    
    %% APG begins
    for iOut = 1:maxOuter
        for iIn = 1:maxInner
            y = C_now+(t_prev/t_now)*(z-C_now) + ((t_prev-1)/t_now)*(C_now-C_prev);
    
            %% line search for y
            s = z-y_prev; 
            s = s(:);
            r = nabla_fxiC(xi,G,z,sigma2,Pfix,a,SEqos)-nabla_fxiC(xi,G,y_prev,sigma2,Pfix,a,SEqos);
            r = r(:);
            stepsize_y = abs( max((s'*s)/(s'*r),(s'*r)/(r'*r)));
    
            Fy = fxiC(xi,G,y,sigma2,Pfix,a,SEqos);
            gradFy = nabla_fxiC(xi,G,y,sigma2,Pfix,a,SEqos);
            
            for iLS = 1:maxLS
                z = projC(Pmax, y + stepsize_y*gradFy);
                Fz = fxiC(xi,G,z,sigma2,Pfix,a,SEqos);
                if Fz >= (Fy+delta*norm(z-y)^2)
                    break
                else
                    stepsize_y = stepsize_y * nu;
                end
            end
            y_prev = y;
    
            %% line search for C
            s = v_prev-C_prev; 
            s = s(:);
            r = nabla_fxiC(xi,G,v_prev,sigma2,Pfix,a,SEqos)-nabla_fxiC(xi,G,C_prev,sigma2,Pfix,a,SEqos);
            r = r(:);
            stepsize_C = abs( max((s'*s)/(s'*r),(s'*r)/(r'*r)));
            
            FC = fxiC(xi,G,C_now,sigma2,Pfix,a,SEqos);
            gradFC = nabla_fxiC(xi,G,C_now,sigma2,Pfix,a,SEqos);
    
            for iLS = 1:maxLS
                v = projC(Pmax, C_now + stepsize_C*gradFC);
                Fv = fxiC(xi,G,v,sigma2,Pfix,a,SEqos);
                if Fv >= (FC+delta*norm(v-C_now)^2)
                    break
                else
                    stepsize_C = stepsize_C * nu;
                end
            end
            v_prev = v;
    
            %%
            C_prev = C_now;
    
            Fz = fxiC(xi,G,z,sigma2,Pfix,a,SEqos);
    
            Fv = fxiC(xi,G,v,sigma2,Pfix,a,SEqos);
    
            if(Fz>=Fv)
                bestobj(m)=Fz;
                C_now = z;
            else
                bestobj(m)=Fv;
                C_now = v;
            end
            
            %%
            t_prev = t_now;
            t_now = (sqrt(4*t_prev^2+1)+1)/2;
    
            
    
            if (iIn>interval)
                tol = (bestobj(m)-bestobj(m-interval))/bestobj(m-interval);
                if (abs(tol)<1e-3) || (iIn == maxInner) % if APG converges, break
                    Penal = PenalC(xi,G,C_now,sigma2,Pfix,a,SEqos);
                    break
                end
            end
    
            m = m+1;
    
        end
        
        if(Penal<1e-4) % if feasible point achieved, break the outer loop
            bestobj(m:end)=[];
            break
        else % if not, increase penalty parameter
            xi = xi*10;
        end
    
    end
    
    
    %% performance
    P = C_now.^2;
end
function kk = f(k,t,y)
global A b c dt n

% Definition of the right-hand side of the ODE discretization

    rhs = @(t,y) -50.0.*(y-cos(t));

    k1 = k(1:n);
    k2 = k(n+1:end);

    kk1 = rhs(t+c(1)*dt,y+dt*(A(1,1)*k1+A(1,2)*k2))-k1;
    kk2 = rhs(t+c(2)*dt,y+dt*(A(2,1)*k1+A(2,2)*k2))-k2;


    kk = [kk1; kk2];
    
end

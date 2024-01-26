clear all; close all;

global A b c dt n

A = zeros(2);
A(1,1) = 1.0/4.0;
A(1,2) = 1.0/4.0 - sqrt(3.0)/6.0;
A(2,1) = 1.0/4.0 + sqrt(3.0)/6.0;
A(2,2) = 1.0/4.0;

b = zeros(2,1);
b(1) = 1./2.;
b(2) = 1./2.;

c = zeros(2,1);
c(1) = 1./2. - sqrt(3.0)/6.0;
c(2) = 1./2. + sqrt(3.0)/6.0;

% initial data
y0 = 0;

% parameters
t0 = 0.0;
Tfinal = 1.5;
N = 150;
dt = (Tfinal-t0)/N;


time = linspace(0,Tfinal,N+1);

%
maxit=100; 
tol=1.e-15; 
h=1.e-15;

t = 0;


sol = zeros(N+1,1);
y = y0;
sol(1) = y;
i = 1;

n = length(y0);

k = zeros(2*n,1);

while (i <= N)
    i = i + 1;

    [k,iters] = cnewton(@f,k,maxit,tol,h, t, y);

    k1 = k(1:n);
    k2 = k(n+1:end);
    y = y + dt*(b(1)*k1 + b(2)*k2);

    sol(i) = y;

    t = time(i);
end

plot(time,sol,'-o')

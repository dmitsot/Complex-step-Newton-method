% Initial guess
x0 = zeros(2,1);

% Parameters used in cnewton
maxit=100; tol=1.e-10; h=1.e-3;

% main 
[x,iters] = cnewton(@f,x0,maxit,tol,h)

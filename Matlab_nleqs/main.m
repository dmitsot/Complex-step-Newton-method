x0 = zeros(2,1);

maxit=100; tol=1.e-10; h=1.e-3;

[x,iters] = cnewton(@f,x0,maxit,tol,h)

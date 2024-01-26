function [xstar, iters] = cnewton(f, x0, maxit, tol, h)

    error = 1.0;
    iters = 0;
    
    x = x0;

    while (iters<=maxit && error >= tol)
        iters = iters + 1;

        b = h*f(x);
        
        % x = gmres(@(u) mv(u,f,x,h),b');
        [b,flag,relres,iter] = gmres(@(u) imag(f(x+1i*h*u)),b);
        x = x - b;

        error = norm(b);
    end

    xstar = x;

end


function [xstar, iters] = cnewton(f, x0, maxit, tol, h, t, y)
global A b c dt

    error = 1.0;
    iters = 0;
    
    x = x0;

    while (iters<=maxit && error >= tol)
        iters = iters + 1;

        r = h*f(x,t,y);
        % x = gmres(@(u) mv(u,f,x,h),b');
        [r,flag,relres,iter] = gmres(@(u) imag(f(x+1i*h*u,t,y)),r);
        x = x - r;
        error = norm(r);
    end

    xstar = x;

end


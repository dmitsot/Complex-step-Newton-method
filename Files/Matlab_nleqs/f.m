function y = f(x)
% Defines the system of equations

    y = zeros(2,1);
    y(1) = x(1)+0.25*x(2)^2-1.25;
    y(2) = 0.25*x(1)^2+x(2)-1.25;
    
end

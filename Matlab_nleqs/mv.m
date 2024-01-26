function z = mv(u,f,x,h)

        z = imag(f(x+1i*h*u));

end
    
      
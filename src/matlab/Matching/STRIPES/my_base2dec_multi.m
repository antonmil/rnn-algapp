function y = my_base2dec_multi(x,base)

y = 0;
fact = 1;
for i=length(x):-1:1
    y = y + x(i)*fact;
    fact = fact*base(i);
end

function v = my_dec2base_multi(x,base)

n = 1;
N = length(base);
v = zeros(1,N);

while x>0
   curr_base = base(end-n+1);
   v(end-n+1) = rem(x,curr_base);
   x = floor(x/curr_base);
   n = n  + 1;
end

v = v + 1;
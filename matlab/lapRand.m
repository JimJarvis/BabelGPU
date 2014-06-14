function y  = lapRand(m, n, mu, sigma)
%LAPRND generate i.i.d. laplacian random number drawn from laplacian distribution
%   with mean mu and standard deviation sigma. 
%   mu      : mean
%   sigma   : standard deviation
%   [m, n]  : the dimension of y.
%   Default mu = 0, sigma = 1. 

if nargin == 2
    mu = 0; sigma = 1;
end

if nargin == 3
    sigma = 1;
end

u = rand(m, n)-0.5;
b = sigma / sqrt(2);
y = mu - b * sign(u).* log(1- 2* abs(u));

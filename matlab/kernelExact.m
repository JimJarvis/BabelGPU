%% kernel: 'gauss' or 'lap'
%
function res = kernelExact(x1, x2, lambda, kernel)

if strcmp(kernel, 'gauss')
    res = exp(-lambda * sumsqr(x1-x2));
elseif strcmp(kernel, 'lap')
    res = exp(-lambda * sum(abs(x1-x2)));
elseif strcmp(kernel, 'cauchy')
    res = prod(1./(1 + lambda * (x1 - x2).^2/2));
end

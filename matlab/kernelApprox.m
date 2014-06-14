%% kernel: 'gauss' or 'lap'
%
function W = kernelApprox(oldDim, newDim, lambda, kernel)

W = zeros(newDim, oldDim + 1);

if strcmp(kernel, 'gauss')
W(1:newDim, 1:oldDim) = normrnd(0, 1, newDim, oldDim) * sqrt(2 * lambda);

elseif strcmp(kernel, 'lap')
W(1:newDim, 1:oldDim) = tan(pi*(rand(newDim, oldDim)-0.5)) * lambda;

elseif strcmp(kernel, 'cauchy')
W(1:newDim, 1:oldDim) = lapRand(newDim, oldDim) * sqrt(lambda);

end
W(1:newDim, oldDim + 1) = rand(newDim, 1) * 2 * pi;
